import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.resnet import conv1x1, conv3x3
from .losses import gen_anomal_map

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
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
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
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        identity = batch
        out = self.conv1(batch)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            identity = self.upsample(batch)
        out += identity
        return self.relu(out)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 512
        self.layer1 = self._make_layer(1, 256, 2)
        self.layer2 = self._make_layer(1, 128, 2)
        self.layer3 = self._make_layer(1, 64, 2)
    def forward(self, x):
        x1 = self.layer1(x) #512x8x8->256x16x16
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return {'layer1': x3, 'layer2': x2, 'layer3': x1}

    def _make_layer(self, blocks, planes, stride):
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
            layers.append(DecoderBlock(self.inplanes, planes, stride, upsample))
        self.inplanes = planes
        return nn.Sequential(*layers)


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


#model
class RevDistill_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        m = resnet18(weights = 'IMAGENET1K_V1')
        return_nodes = {'layer1':'layer1', 'layer2':'layer2', 'layer3':'layer3'}
        self.encoder = create_feature_extractor(m, return_nodes=return_nodes)
        self.bottleneck = Bottleneck()
        self.ocbe = m.layer4
        self.decoder = Decoder()
        self.anomap_gen = AnomalyMapGenerator()
        self.tau = config['loss']['tau']

        for param in self.encoder.parameters():
            param.requires_grad = False

        #for param in self.ocbe.parameters():
            #param.requires_grad = False

        self.proj = nn.ModuleDict({
            'layer1': nn.Conv2d(64, 256, 1),
            'layer2': nn.Conv2d(128, 256, 1),
            'layer3': nn.Conv2d(256, 256, 1)
        })
        self.proj = self.proj.cuda()

    def forward(self, x, train=False):
        feats = self.encoder(x)
        f = self.bottleneck(feats)
        f = self.ocbe(f)
        f = self.decoder(f)
        if train:
            return feats, f
        else:
            anomap = self.anomap_gen(feats, f, x.shape, self.proj)
            #anomap = gen_anomal_map(feats, f, self.proj, x.shape, self.tau)
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

