import torch
import torch.nn as nn
from .superpixel import *
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=None, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if self.alpha is not None:
            self.alpha = torch.Tensor([alpha, 1-alpha])
        self.size_average = size_average
    def forward(self, input, target):
        target = target.view(-1,1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at
        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def prior_cross_entropy(batch_preds, batch_regions):
    batch_size = len(batch_preds)
    batch_loss = 0.
    for i in range(batch_size):
        preds = batch_preds[i] #(N, 2)
        regions = batch_regions[i] #(h, w)
        N = regions.unique().shape[0]
        labels = make_region_labels(regions)
        labels = labels.long().to(preds.device) #(N, )
        weights = torch.tensor([1., N - 1.]).to(preds.device)
        # occurrence weighting
        cross_entropy = nn.CrossEntropyLoss(
            reduction='sum',
            weight=weights)
        loss = cross_entropy(preds, labels)
        batch_loss += loss
    batch_loss /= batch_size
    return batch_loss


