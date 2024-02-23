import torch
import torch.nn as nn
from open_clip import create_model_and_transforms
from .superpixel import *
from .losses import *
from .prompt import *
from torch_geometric.nn import GCNConv, GATConv

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
            self.gnn1 = GATConv(640, 64, heads = self.heads)
            self.gnn2 = GATConv(64, 64, heads = self.heads)
            self.gnn3 = GATConv(64, 640, heads = self.heads)
    def forward(self, x, edges):
        x = F.relu(self.gnn1(x, edges.T))
        x = F.relu(self.gnn2(x, edges.T))
        x = self.gnn3(x, edges.T)
        return x

class RegionClipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.level = config['gnn']['level']
        self.clip, _, self._transform = create_model_and_transforms(
            model_name=config['clip']['model_name'],
            pretrained=config['clip']['pretrained']
        )
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip = self.clip.to('cuda')
        self.gnn = GNN(config).to('cuda')
        self.prompt = Learned_Prompt(config, self.clip).to('cuda')
        self.get_text_embs()
    def forward(self, x):
        x = x.cuda()
        batch_size = x.shape[0]
        batch_region_embs, batch_edges, batch_regions = self.get_region_embs(x) #[(N, d), ...], [[...,]]
        text_embs = F.normalize(self.text_embs).to(x.device) #(2, 640)
        temp = self.config['clip']['temp']
        batch_preds = []
        for i in range(batch_size):
            region_embs = batch_region_embs[i].to(x.device) #(N, d)
            edges = batch_edges[i].to(x.device)
            region_nodes = self.gnn(region_embs, edges)
            if self.level == 'node':
                region_nodes = F.normalize(region_nodes, dim=1)
                pred = region_nodes @ text_embs.T/temp #(N, 2)
            else:
                max_node, _ = region_nodes.max(dim=0) #(d,)
                max_node = F.normalize(max_node, dim=0)
                pred = max_node @ text_embs.T/temp #(2, )
            batch_preds.append(pred)
        return batch_preds, batch_regions

    def get_text_embs(self):
        normal_embs, anormal_embs = self.prompt()
        self._text_embs = torch.cat([normal_embs, anormal_embs], axis=0) #(2, d)

    def get_region_embs(self, x):
        batch_size = x.shape[0]
        batch_region_embs = []
        batch_regions = []
        batch_edges = []
        for i in range(batch_size):
            #regions: (h, w), edges:[...]
            x_i = x[i]
            regions, edges = super_pixel_graph_construct(
                x_i, **self.config['superpixel'])
            x_i = region_sampling(x_i, regions)  # (N, 3, h, w)
            region_embs = self.clip.encode_image(x_i)  # (N, d)
            region_embs = region_embs.view(-1, region_embs.shape[-1])
            batch_region_embs.append(region_embs) #(N, d)
            batch_regions.append(regions)
            batch_edges.append(edges)
        return batch_region_embs, batch_edges, batch_regions

    @property
    def text_embs(self):
        return self._text_embs