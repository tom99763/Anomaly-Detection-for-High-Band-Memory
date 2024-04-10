import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.resnet import conv1x1, conv3x3
import numpy as np

#bottleneck
class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj1 = nn.Sequential(
            conv3x3(64, 128, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            conv3x3(128, 256, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.proj2 = nn.Sequential(
            conv3x3(128, 256, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fuse = conv1x1(256*3, 256)
    def forward(self, feats):
        feats_ = []
        for k, v in feats.items():
            if k=='layer1':
                v = self.proj1(v)
            elif k=='layer2':
                v = self.proj2(v)
            feats_.append(v)
        feats_ = torch.cat(feats_, dim=1)
        feats_ = self.fuse(feats_)
        return feats_

#decoder
class DecoderBlock(nn.Module):
    expansion = 1.
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample = None,
        norm_layer = 'bn',
    ) -> None:
        super().__init__()
        self.norm_layer = norm_layer
        if self.norm_layer  == 'bn':
            norm_layer = nn.BatchNorm2d
        elif self.norm_layer == 'adain':
            norm_layer = Att_AdaIn
        # Both self.conv1 and self.downsample layers downsample the input when stride != 2
        if stride == 2:
            self.conv1 = nn.ConvTranspose2d(
                inplanes,
                planes,
                kernel_size=2,
                stride=stride,
                bias=False,
            )
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride
    def forward(self, x, y=None):
        identity = x
        if self.norm_layer == 'bn':
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
        elif self.norm_layer == 'adain':
            out = self.conv1(x)
            out = self.bn1(out, y)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out, y)
        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity
        return out

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_layer = config['norm']['norm_layer']
        self.inplanes = 512
        self.layer1 = self._make_layer(1, 256, 2, 'bn')
        self.layer2 = self._make_layer(1, 128, 2, 'bn')
        self.layer3 = self._make_layer(1, 64, 2, 'bn')
        self.layer4 = self._make_layer(1, 64, 2, 'bn')
        self.layer5 = self._make_layer(1, 64, 2, 'bn')
        self.layer6 = nn.Conv2d(64, 3, 7, 1, 3)
    def forward(self, x, y=None):
        #layer3
        x1 = x
        y1 = y['layer3']
        for layer in self.layer1:
            x1 = layer(x1, y1)
        #layer2
        x2 = x1
        y2 = y['layer2']
        for layer in self.layer2:
            x2 = layer(x2, y2)
        #layer1
        x3 = x2
        y3 = y['layer1']
        for layer in self.layer3:
            x3 = layer(x3, y3)
        #layer0
        x4 = x3
        for layer in self.layer4:
            x4 = layer(x4)
        x5 = x4
        for layer in self.layer5:
            x5 = layer(x5)
        x6 = self.layer6(x5)
        return x6

    def _make_layer(self, blocks, planes, stride, norm_layer):
        upsample = nn.Sequential(
            nn.ConvTranspose2d(
                self.inplanes,
                planes,
                kernel_size=2,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(planes),
        )
        layers = []
        for i in range(blocks):
            layers.append(
                DecoderBlock(self.inplanes, planes, stride, upsample, norm_layer=norm_layer).cuda())
        self.inplanes = planes
        layers.append(
            DecoderBlock(self.inplanes, planes, norm_layer=norm_layer).cuda())
        return layers


#anomaly generator
class AnomalyMapGenerator(nn.Module):
    def __init__(self):
        super().__init__()
    def compute_layer_map(self, ft, fs, img_size, proj_layer):
        ft = F.normalize(proj_layer(ft))
        fs = F.normalize(proj_layer(fs))
        #layer_map = 0.5 * torch.norm(ft - fs, p=2, dim=-3, keepdim=True) ** 2
        layer_map = 1-torch.sum(ft * fs, dim=1, keepdim=True)
        return F.interpolate(layer_map,
                             size=img_size,
                             align_corners=False,
                             mode="bilinear",
                             )
    def forward(self, ft, fs, img_size, proj_layers):
        b, _, h, w = img_size
        anomaly_map = torch.ones(b, 1, h, w).cuda()
        for layer in ft:
            layer_map = self.compute_layer_map(ft[layer], fs[layer], (h, w), proj_layers[layer])
            anomaly_map *= layer_map
        return anomaly_map

#adain
class Att_AdaIn(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.inst_norm = nn.Identity()
        self.to_q = conv1x1(inplanes, inplanes)
        self.to_k = conv1x1(inplanes, inplanes)
        self.to_v = conv1x1(inplanes, inplanes)
        self.out = conv1x1(inplanes, inplanes)
        self.scale = torch.sqrt(torch.tensor(inplanes, dtype=torch.float32))
    def forward(self, x, y):
        '''
        :param x: content
        :param y: style
        '''
        b, c, h, w = x.shape
        x_norm = self.inst_norm(x) #b,c,h,w
        y_norm = self.inst_norm(y)
        q = self.to_q(x_norm).flatten(2) #b,c,hw
        k = self.to_k(y_norm).flatten(2) #b,c,hw
        v = self.to_v(y).flatten(2) #b,c,hw
        A = (torch.einsum('bck, bcv->bkv', q, k)/self.scale).softmax(dim=-1)
        z = torch.einsum('blk, bck->bcl', A, v) #b,c,hw
        z = z.view(-1, c, h, w)
        z = self.out(z)
        return x + z



#model
class RevDistill_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        m = resnet18(weights = 'IMAGENET1K_V1')
        return_nodes = {'layer1':'layer1', 'layer2':'layer2', 'layer3':'layer3'}
        self.encoder = create_feature_extractor(m, return_nodes=return_nodes).cuda()
        self.bottleneck = Bottleneck().cuda()
        self.ocbe = m.layer4
        self.decoder = Decoder(config).cuda()
        self.anomap_gen = AnomalyMapGenerator()
        self.tau = config['loss']['tau']

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.proj = nn.ModuleDict({
            'layer0': nn.Conv2d(3, 256, 1),
            'layer1': nn.Conv2d(64, 256, 1),
            'layer2': nn.Conv2d(128, 256, 1),
            'layer3': nn.Conv2d(256, 256, 1)
        })
        self.proj = self.proj.cuda()
        self.norm_layer = config['norm']['norm_layer']

    def forward(self, x, train=False):
        ft = self.encoder(x)
        fs = self.bottleneck(ft)
        fs = self.ocbe(fs)
        xs = self.decoder(fs, ft)
        fs = self.encoder(xs)
        ft['layer0'] = x
        fs['layer0'] = xs

        if train:
            return ft, fs
        else:
            anomap = self.anomap_gen(ft, fs, x.shape, self.proj)
            return anomap

    def train_mode(self):
        self.encoder.eval()
        self.ocbe.train()
        self.bottleneck.train()
        self.decoder.train()

    def eval_mode(self):
        self.encoder.eval()
        self.ocbe.eval()
        self.bottleneck.eval()
        self.decoder.eval()



'''
if __name__ == '__main__':
    config = {'mask':{'ratio':0.75, 'num_seeds':10},
              'norm':{'norm_layer':'adain'},
              'loss':{'tau':0.07, 'num_negs':256}
              }
    m = RevDistill_Model(config).cuda()
    x = torch.randn(4, 3, 256, 256).cuda()
    o = m(x)
    print(o.shape)
'''


