import torch
import torch.nn as nn
from .superpixel import *
import torch.nn.functional as F
import numpy as np

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

def vitst_loss(ft, fs):
    ft = F.normalize(ft, dim=1)
    fs = F.normalize(fs, dim=1)
    h, w  = ft.shape[2:]
    loss = (0.5 / (h * w)) * mse(ft, fs)
    return loss


class Patch_infonce(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_negs = config['loss']['num_negs']
        self.tau = config['loss']['tau']

        # project layers
        self.ce = nn.CrossEntropyLoss()
        self.ce_anomap = nn.CrossEntropyLoss(reduction='none')
    def forward(self, ft, fs, proj_layers):
        loss = torch.tensor(0.).cuda()
        for layer in ft:
            fs_ = fs[layer] #(b, d, h, w)
            ft_ = ft[layer]  #(b, d, h, w)
            fs_ = proj_layers[layer](fs_).flatten(2) #(b, d, n)
            ft_ = proj_layers[layer](ft_).flatten(2) #(b, d, n)

            #patch sampling
            length = ft_.shape[-1]
            num_negs = min(length, self.num_negs)
            idx = np.random.choice(torch.arange(0, length), size = num_negs, replace=False)
            idx = torch.tensor(sorted(idx)).cuda()
            fs_ = F.normalize(fs_[:, :, idx], dim=-1) #(b, d, n)
            ft_ = F.normalize(ft_[:, :, idx], dim=-1).detach() #(b, d, n)

            #infonce
            l_pos = (fs_*ft_).sum(dim=1)[..., None] #(b, n, 1)
            l_neg = torch.bmm(fs_.permute(0, 2, 1), ft_) #(b, n, n)
            idt_m = torch.eye(num_negs)[None, :, :].bool().cuda()
            l_neg.masked_fill_(idt_m, -float('inf'))
            logits = torch.cat([l_pos, l_neg], dim=2)/self.tau #(b, n, n+1)
            preds = logits.flatten(0, 1)
            bn, _ = preds.shape
            targets = torch.zeros(bn, dtype=torch.long).cuda()
            loss += self.ce(preds, targets)
        loss = loss/len(ft)
        return loss

def gen_anomal_map(ft, fs, proj_layers, img_size, tau):
    ce_anomap = nn.CrossEntropyLoss(reduction='none')
    b, _, h, w = img_size
    loss = torch.ones(b, 1, h, w).cuda()
    for layer in ft:
        fs_ = fs[layer]  # (b, d, h, w)
        ft_ = ft[layer]  # (b, d, h, w)
        _, _, h_, w_ = ft_.shape
        fs_ = proj_layers[layer](fs_).flatten(2)  # (b, d, n)
        ft_ = proj_layers[layer](ft_).flatten(2)  # (b, d, n)

        # patch sampling
        length = ft_.shape[-1]
        fs_ = F.normalize(fs_, dim=-1)  # (b, d, n)
        ft_ = F.normalize(ft_, dim=-1).detach()  # (b, d, n)

        # infonce
        l_pos = (fs_ * ft_).sum(dim=1)[..., None]  # (b, n, 1)
        l_neg = torch.bmm(fs_.permute(0, 2, 1), ft_)  # (b, n, n)
        idt_m = torch.eye(length)[None, :, :].bool().cuda()
        l_neg.masked_fill_(idt_m, -float('inf'))
        logits = torch.cat([l_pos, l_neg], dim=2) / tau  # (b, n, n+1)
        preds = logits.flatten(0, 1)
        bn, _ = preds.shape
        targets = torch.zeros(bn, dtype=torch.long).cuda()
        loss_ = ce_anomap(preds, targets)  # (b, n)
        loss_ = loss_.view(b, 1, h_, w_)/(h_*w_ + 1)
        loss_ = F.interpolate(loss_,
                              size=(h, w),
                              align_corners=False,
                              mode="bilinear",
                              )
        loss *= loss_
    return loss

