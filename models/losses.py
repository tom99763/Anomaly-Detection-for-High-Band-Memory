import torch
import torch.nn as nn
from .superpixel import *
import torch.nn.functional as F

cross_entropy = nn.CrossEntropyLoss()

mse = nn.MSELoss()


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
            batch_anorm_idx, text_embs, temp=0.07, step = 0.):
    batch_size = len(batch_anorm_idx)
    text_embs = F.normalize(text_embs, dim=-1)
    loss = torch.tensor(0.).cuda()
    for i in range(batch_size):
        region_embs = batch_region_embs[i] #(N, d)
        region_nodes = batch_region_nodes[i] #(N, d)
        region_emb_preds = batch_region_emb_preds[i] #(N, 2)
        region_node_preds = batch_region_node_preds[i] #(N, 2)
        anorm_idx = batch_anorm_idx[i]
        #mask
        N = region_embs.shape[0]
        label = torch.zeros((N, )).cuda()
        label[anorm_idx] = 1.
        label_ohe = F.one_hot(label.long(), 2).float() #(N, 2)
        mask = label_ohe @ label_ohe.T #(N, N)
        mask_diag = torch.eye(N).cuda()
        mask = mask - mask_diag

        #nodes similarity
        region_nodes = F.normalize(region_nodes, dim=-1)
        #region_embs = F.normalize(region_embs, dim=-1)
        sim_node = region_nodes @ region_nodes.T #(N, N)
        #sim_node = region_nodes @ region_embs.T
        #sim_max = torch.maximum(sim_node - sim_emb, torch.zeros_like(sim_node).cuda())
        #print(sim_max.sum())
        sim_node = torch.exp(sim_node / temp) * (1-mask_diag)
        sim_node_pos = sim_node * mask


        #text similarity
        sim_text = region_nodes @ text_embs.T
        sim_text = torch.exp(sim_text/temp)
        sim_text = torch.gather(
            sim_text, 1, torch.tensor(label[:, None].clone().detach(), dtype=torch.int64))

        #loss
        if step<0: #warm-up
            numerator = sim_text
            loss_ = -torch.log(numerator)
        else:
            numerator = sim_text + sim_node_pos.sum(dim=1)
            denomerator = torch.cat([sim_text, sim_node], dim=1).sum(dim=1)
            loss_ = -torch.log(numerator / denomerator)
        #else
        #numerator = sim_text
        #loss_ = -torch.log(numerator)
        loss_ = loss_.mean()
        loss += loss_
    loss = loss/batch_size
    return loss

def cross_entropy_loss(batch_pred, batch_unlabeled_idx):
    batch_size = len(batch_pred)
    l_ce = torch.tensor(0.).cuda()
    for i in range(batch_size):
        unlabeled_idx = batch_unlabeled_idx[i].tolist()
        pred = batch_pred[i]
        N=pred.shape[0]
        idx = torch.arange(0, N).tolist()
        labeled_idx = list(set(idx)-set(unlabeled_idx))
        labeled_pred = pred[labeled_idx]
        label = torch.zeros(N,).cuda().long()
        l_ce += cross_entropy(labeled_pred, label)
    return l_ce/batch_size

def div_consistency_loss(batch_h, batch_h_aug,
                     batch_pred, batch_pred_aug,
                     batch_unlabeled_idx):
    batch_size = len(batch_h)
    l_c = torch.tensor(0.).cuda()
    l_div = torch.tensor(0.).cuda()
    for i in range(batch_size):
        h = batch_h[i]
        h_aug = batch_h_aug[i]
        pred = batch_pred[i]
        pred_aug = batch_pred_aug[i]
        unlabeled_idx = batch_unlabeled_idx[i]

        #consistency
        unlabeled_pred = pred[unlabeled_idx]
        unlabeled_pred_aug = pred_aug[unlabeled_idx]
        label = unlabeled_pred.argmax(dim=-1)
        label_aug = unlabeled_pred_aug.argmax(dim=-1)
        c_idx = label == label_aug
        if c_idx.float().sum()==0.:
            continue
        logits_aug = unlabeled_pred_aug[c_idx]
        label = label.long()
        l_c += cross_entropy(logits_aug, label)

        #diversity
        unlabeled_h = h[unlabeled_idx]
        unlabeled_h = unlabeled_h[c_idx]
        unlabeled_h_aug = h_aug[unlabeled_idx]
        unlabeled_h_aug = unlabeled_h_aug[c_idx]
        l_div = l_div - mse(unlabeled_h, unlabeled_h_aug)
    return l_c/batch_size, l_div/batch_size





