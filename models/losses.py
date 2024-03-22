import torch
import torch.nn as nn
from .superpixel import *
import torch.nn.functional as F

cross_entropy = nn.CrossEntropyLoss()

mse = nn.MSELoss(reduction="sum")

def stfpm_loss(ft, fs):
    loss = torch.tensor(0.).cuda()
    for layer in ft:
        ftl = F.normalize(ft[layer], dim=1)
        fsl = F.normalize(fs[layer], dim=1)
        h, w = ftl.shape[2:]
        loss+=(0.5 / (h*w)) * mse(ftl, fsl)
    return loss







