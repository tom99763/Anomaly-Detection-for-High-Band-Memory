import torch
import torch.nn as nn
from .superpixel import *

def prior_cross_entropy(batch_preds, batch_regions):
    batch_size = len(batch_preds)
    batch_loss = 0.
    for i in range(batch_size):
        preds = batch_preds[i] #(N, 2)
        regions = batch_regions[i] #(h, w)
        N = regions.unique().shape[0]
        labels = make_region_labels(regions)
        labels = labels.long().to(preds.device) #(N, )
        weights = torch.tensor([1., N-1.]).to(preds.device)
        #occurrence weighting
        cross_entropy = nn.CrossEntropyLoss(
            reduction = 'sum',
            weight=weights)
        batch_loss += cross_entropy(preds, labels)
    batch_loss /= batch_size
    return batch_loss


