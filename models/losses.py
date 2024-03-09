import torch
import torch.nn as nn
from .superpixel import *
import torch.nn.functional as F

cross_entropy = nn.CrossEntropyLoss()

mse = nn.MSELoss()

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

def region_wise_cross_entropy(batch_preds):
    batch_size = len(batch_preds)
    loss = torch.tensor(0.).cuda()
    for i in range(batch_size):
        preds = batch_preds[i] #(N, 2)
        label = torch.zeros((preds.shape[0],), dtype=torch.long).cuda()
        loss += cross_entropy(preds, label)
    return loss/batch_size

def contrastive_loss(batch_region_embs, batch_region_nodes,
        batch_region_emb_preds, batch_region_node_preds,
            batch_anorm_idx, text_embs, temp=0.07):
    batch_size = len(batch_anorm_idx)
    text_embs = F.normalize(text_embs, dim=-1)
    loss = torch.tensor(0.).cuda()
    for i in range(batch_size):
        region_embs = batch_region_embs[i] #(N, d)
        region_nodes = batch_region_nodes[i] #(N, d)
        region_emb_preds = batch_region_emb_preds[i] #(N, 2)
        region_node_preds = batch_region_node_preds[i] #(N, 2)
        anorm_idx = batch_anorm_idx[i]
        N = region_embs.shape[0]
        label = torch.zeros((N, )).cuda()
        label[anorm_idx] = 1.
        label_ohe = F.one_hot(label.long(), 2).float() #(N, 2)
        mask = label_ohe @ label_ohe.T #(N, N)
        mask_diag = torch.eye(N).cuda()
        mask = mask - mask_diag

        #nodes similarity
        region_nodes = F.normalize(region_nodes, dim=-1)
        sim_node = region_nodes @ region_nodes.T #(N, N)
        sim_node = torch.exp(sim_node/temp)
        sim_node = (1-mask_diag) * sim_node
        sim_node_pos = sim_node * mask


        #text similarity
        sim_text = torch.exp(region_nodes @ text_embs.T/temp)
        sim_text = torch.gather(
            sim_text, 1, torch.tensor(label[:, None].clone().detach(), dtype=torch.int64))

        #loss
        numerator = sim_text + sim_node_pos.sum(dim=1) #(N, )
        denomerator = torch.cat([sim_text, sim_node], dim=1).sum(dim=1) #(N, )

        loss_ = -torch.log(numerator / denomerator)
        loss_ = loss_.mean()

        loss += loss_
    loss = loss/batch_size
    return loss


def margin_contrastive_loss(batch_region_embs, batch_region_nodes,
        batch_region_emb_preds, batch_region_node_preds,
            batch_anorm_idx, text_embs, temp=0.07):
    batch_size = len(batch_anorm_idx)
    text_embs = F.normalize(text_embs, dim=-1)
    loss = torch.tensor(0.).cuda()
    for i in range(batch_size):
        region_embs = batch_region_embs[i] #(N, d)
        region_nodes = batch_region_nodes[i] #(N, d)
        region_emb_preds = batch_region_emb_preds[i] #(N, 2)
        region_node_preds = batch_region_node_preds[i] #(N, 2)
        anorm_idx = batch_anorm_idx[i]
        N = region_embs.shape[0]
        label = torch.zeros((N, )).cuda()
        label[anorm_idx] = 1.
        label_ohe = F.one_hot(label.long(), 2).float() #(N, 2)
        mask = label_ohe @ label_ohe.T #(N, N)
        mask_diag = torch.eye(N).cuda()
        mask = mask - mask_diag

        #nodes similarity
        region_nodes = F.normalize(region_nodes, dim=-1)
        sim_node = region_nodes @ region_nodes.T #(N, N)
        sim_emb = region_embs @ region_embs.T
        sim_min = torch.minimum(sim_node - sim_emb, torch.zeros_like(sim_node).cuda())
        sim_max = torch.maximum(sim_node - sim_emb, torch.zeros_like(sim_node).cuda())
        sim_min = torch.exp(sim_min/temp) * mask
        sim_max = torch.exp(sim_max / temp) * (1-mask_diag)


        #text similarity
        sim_text = region_nodes @ text_embs.T
        sim_text_prev = region_embs @ text_embs.T
        sim_text_min = torch.minimum(sim_text-sim_text_prev, torch.zeros_like(sim_text).cuda())
        sim_text_min = torch.exp(sim_text_min/temp)
        sim_text_min = torch.gather(
            sim_text_min, 1, torch.tensor(label[:, None].clone().detach(), dtype=torch.int64))

        #loss
        numerator = sim_text_min + sim_min.sum(dim=1)
        denomerator = torch.cat([sim_text_min, sim_max], dim=1).sum(dim=1)

        loss_ = -torch.log(numerator / denomerator)
        loss_ = loss_.mean()

        loss += loss_
    loss = loss/batch_size
    return loss

