import torch
import torch.nn as nn
from open_clip import create_model_and_transforms
from .superpixel import *
from .losses import *
from .prompt import *
from torch_geometric.nn import GCNConv, GATConv

class GNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return

class RegionClipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.clip, _, self._transform = create_model_and_transforms(**config['clip'])
        for param in self.clip.parameters():
            param.requires_grad = False
        self.gnn = GNN()
        self.prompt = Learned_Prompt(config, self.clip)
        self.get_text_embs()

    def forward(self):
        return

    def get_text_embs(self):
        normal_embs, anormal_embs = self.prompt()
        self._text_embs = torch.cat([normal_embs, anormal_embs], axis=0) #(2, d)

    def get_region_embs(self, x):
        batch_size = x.shape[0]
        batch_region_embs = []
        batch_edges = []
        for i in range(batch_size):
            #regions: (h, w), edges:[...]
            x_i = x[i]
            regions, edges = super_pixel_graph_construct(
                x_i, **self.config['superpixel'])
            x_i = region_sampling(x_i, regions)  # (N, 3, h, w)
            region_embs = self.clip.encode_image(x_i)  # (N, d)
            region_embs = region_embs.view(-1, region_embs.shape[-1])
            batch_region_embs.append(region_embs) #(N, 640)
            batch_edges.append(batch_edges)
        return batch_region_embs, batch_edges

    @property
    def text_embs(self):
        return self._text_embs