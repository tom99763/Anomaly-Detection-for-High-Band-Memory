import torch
import torch.nn as nn

def prior_cross_entropy(batch_preds, batch_regions):
    batch_size = len(batch_preds)
    batch_loss = 0.
    for i in range(batch_size):
        preds = batch_preds[i] #(N, 2)
        regions = batch_regions[i] #(h, w)
        h, w = regions.shape
        N = preds.shape[0]
        target_region = regions[h//2, w//2]
        labels = torch.zeros((N,))
        labels[target_region] = 1
        labels = labels.long().to(preds.device) #(N, )
        #occurrence weighting
        cross_entropy = nn.CrossEntropyLoss(
            reduction = 'sum',
            weight=torch.tensor([1., N-1.]).to(preds.device))
        batch_loss += cross_entropy(preds, labels)
    batch_loss /= batch_size
    return batch_loss


