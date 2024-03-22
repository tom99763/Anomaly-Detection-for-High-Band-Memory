import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import *
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor

class AnomalyMapGenerator(nn.Module):
    def __init__(self):
        super().__init__()
    def compute_layer_map(self, ft, fs, img_size):
        ft = F.normalize(ft)
        fs = F.normalize(fs)
        layer_map = 0.5 * torch.norm(ft - fs, p=2, dim=-3, keepdim=True) ** 2
        return F.interpolate(layer_map,
                             size=img_size,
                             align_corners=False,
                             mode="bilinear",
                             )
    def forward(self, ft, fs, img_size):
        b, _, h, w = img_size
        anomaly_map = torch.ones(b, 1, h, w).cuda()
        for layer in ft:
            layer_map = self.compute_layer_map(ft[layer], fs[layer], (h, w))
            anomaly_map *= layer_map

        return anomaly_map


class STFPM_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        return_nodes = {'layer1':'layer1', 'layer2':'layer2', 'layer3': 'layer3'}
        resnet_teacher = resnet18(weights = 'IMAGENET1K_V1')
        self.teacher = create_feature_extractor(resnet_teacher, return_nodes=return_nodes).cuda()
        resnet_student = resnet18(weights = None)
        self.student =  create_feature_extractor(resnet_student, return_nodes=return_nodes).cuda()

        for param in self.teacher.parameters():
            param.requires_grad=False

        self.anomap = AnomalyMapGenerator()
        self.teacher.eval()

    def forward(self, x, train=False):
        x = x.cuda()
        ft = self.teacher(x)
        fs = self.student(x)
        if train:
            return ft, fs
        else:
            return self.anomap(ft, fs, x.shape)
