
'''
#generates unlabeld nodes
unlabeled Regions : Randomly_Augment(p=0.5)(I)

#graph construction
G={V,E,X} #0.5 labeled; 0.5 unlabeled

#feature
h = g(X)
h' = g(NoiseInjection(X))

#Consistency
C = I(p(X')=p(X))
#Diversity
D = ||X-X'||
'''

import torch
import torch.nn as nn
from open_clip import create_model_and_transforms
from .superpixel import *
from .losses import *
from .prompt import *
from torch_geometric.nn import GATConv, Sequential, GCNConv
from torch_geometric.nn.pool import SAGPooling


class GNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        gnn_type = config['gnn']['gnn_type']
        self.heads = config['gnn']['heads']
        if gnn_type == 'GCN':
            self.gnn1 = GCNConv(640, 64)
            self.gnn2 = GCNConv(64, 64)
            self.gnn3 = GCNConv(64, 640)

        elif gnn_type == 'GAT':
            self.gnn1 = GATConv(640, 64, heads = self.heads, concat=False)
            self.gnn2 = GATConv(64, 64, heads = self.heads, concat=False)
            self.gnn3 = GATConv(64, 640, heads = self.heads, concat=False)
    def forward(self, x, edges):
        x = F.relu(self.gnn1(x, edges.T))
        x = F.relu(self.gnn2(x, edges.T))
        x = self.gnn3(x, edges.T)
        return x



class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(640, 64),
            nn.ReLU(),
            nn.Linear(64, 640)
        )
    def forward(self, x):
        return self.layers(x)


class DimMasking(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = NN()
        self.drop_ratio = config['loss']['drop_ratio']
        self.temp = 0.07
    def forward(self, x):
        h = self.attn(x)
        d = h.shape[-1]
        hhat = torch.zeros_like(h).cuda()
        for i in range(int(d*self.drop_ratio)):
            m= 1-hhat
            mhat = torch.log(m+1e-7)
            y=(-h+mhat)/self.temp
            yhat = y.softmax(dim=-1)
            hhat = hhat + yhat * m
        self.mask = (1-hhat)
        return self.mask * x


class RegionClipSemiModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pad_green = self.config['pad']['pad_green']
        self.net_type = config['gnn']['net_type']
        self.clip, _, self._transform = create_model_and_transforms(
            model_name=config['clip']['model_name'],
            pretrained=config['clip']['pretrained']
        )
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip = self.clip.to('cuda')
        if self.net_type == 'gnn':
            self.gnn = GNN(config).to('cuda')
        elif self.net_type == 'linear':
            self.nn = NN().to('cuda')

        #self.noise_inject = NoiseInjection()
        #self.noise_inject = DimMasking(config)
        self.learned_object = Learned_Object(self.clip).cuda()

    def forward(self, x, pad_green=False):
        x = x.cuda()
        batch_size = x.shape[0]
        batch_region_embs, batch_edges, batch_regions = self.get_region_embs(x, pad_green)

        temp = self.config['clip']['temp']
        # forward
        batch_region_node_preds = []
        batch_region_nodes = []
        batch_text_embs = []
        for i in range(batch_size):
            # no augment
            region_embs = batch_region_embs[i].to(x.device)  # (N, d)
            edges = batch_edges[i].to(x.device)
            if self.net_type == 'gnn':
                region_nodes = self.gnn(region_embs, edges)
            elif self.net_type == 'linear':
                region_nodes = self.nn(region_embs)
            batch_region_nodes.append(region_nodes)

            #conditional text
            region_nodes, _ = region_nodes.max(dim=0, keepdim=True)  # (d, )
            #text_embs = self.learned_object(region_nodes) #(N, 2, d)
            text_embs = self.learned_object(region_nodes)
            batch_text_embs.append(text_embs)
            text_embs = F.normalize(text_embs).to(x.device) #(N, 2, d)

            # pred
            #region_max_node, _ = region_nodes.max(dim=0)  # (d, )
            region_nodes = F.normalize(region_nodes) #(N, d)
            #node_pred = torch.einsum('bkd, bd->bk', text_embs, region_nodes) / temp  # (1, 2)
            node_pred = (region_nodes[:, None] * text_embs).sum(dim=-1)/temp #(N, 2)
            idx = node_pred[:, 1].argmax(dim=-1)
            node_pred = node_pred[idx][None, :] #maximum search
            #node_pred = node_pred.mean(dim=0, keepdim=True) #(1, 2)
            batch_region_node_preds.append(node_pred)
        batch_region_node_preds = torch.cat(batch_region_node_preds, dim=0) #(b, 2)
        return batch_region_node_preds, batch_region_nodes, batch_text_embs, batch_regions

    def get_region_embs(self, x, pad_green=False):
        batch_size = x.shape[0]
        batch_region_embs = []
        batch_regions = []
        batch_edges = []
        for i in range(batch_size):
            # regions: (h, w), edges:[...]
            x_i = x[i]
            regions, edges = super_pixel_graph_construct(
                x_i, **self.config['superpixel'])
            x_i = region_sampling(x_i, regions, pad_green)  # (N, 3, h, w)
            region_embs = self.clip.encode_image(x_i)  # (N, d)
            region_embs = region_embs.view(-1, region_embs.shape[-1])
            batch_region_embs.append(region_embs)  # (N, d)
            batch_regions.append(regions)
            batch_edges.append(edges)
        return batch_region_embs, batch_edges, batch_regions




