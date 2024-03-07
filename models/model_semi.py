import torch
import torch.nn as nn
from open_clip import create_model_and_transforms
from .superpixel import *
from .losses import *
from .prompt import *
from .hand_craft_prompt import *
from torch_geometric.nn import GCNConv, GATConv
from open_clip.tokenizer import tokenize

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

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(640, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 640)
        )
    def forward(self, x):
        return self.layers(x)


class RegionClipSemiModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.level = config['gnn']['level']
        self.linear_probe = config['prompt']['linear_probe']
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

        self.get_text_embs()

    def forward(self, x, pad_green=False):
        x = x.cuda()
        batch_size = x.shape[0]
        batch_region_embs, batch_edges, batch_regions = self.get_region_embs(x, pad_green)

        text_embs = F.normalize(self.text_embs).to(x.device) #(2, 640)
        temp = self.config['clip']['temp']

        batch_region_node_preds = []
        batch_region_emb_preds = []
        batch_region_nodes = []

        for i in range(batch_size):
            region_embs = batch_region_embs[i].to(x.device) #(N, d)
            edges = batch_edges[i].to(x.device)
            if self.net_type == 'gnn':
                region_nodes = self.gnn(region_embs, edges)
            elif self.net_type == 'linear':
                region_nodes = self.nn(region_embs)
            else:
                region_nodes = region_embs

            batch_region_nodes.append(region_nodes)

            #emb prediction
            region_embs = F.normalize(region_embs, dim=1)
            emb_pred = region_embs @ text_embs.T / temp  # (N, 2)

            #node prediction
            region_nodes = F.normalize(region_nodes, dim=1)
            node_pred = region_nodes @ text_embs.T / temp  # (N, 2)

            batch_region_node_preds.append(node_pred)
            batch_region_emb_preds.append(emb_pred)

        return batch_region_embs, batch_region_nodes, \
               batch_region_emb_preds, batch_region_node_preds

    def get_text_embs(self):
        class_name = self.config['clip']['class_name']
        norm_prompts, anorm_prompts = create_prompt_ensemble(class_name)
        mean_norm_prompt = []
        mean_anorm_prompt = []
        for prompt in norm_prompts:
            text_token = tokenize(prompt).to('cuda')
            text_emb = self.clip.encode_text(text_token) #(1, 640)
            mean_norm_prompt.append(text_emb)
        for prompt in anorm_prompts:
            text_token = tokenize(prompt).to('cuda')
            text_emb = self.clip.encode_text(text_token)
            mean_anorm_prompt.append(text_emb)
        mean_norm_prompt = torch.cat(mean_norm_prompt, dim=0).mean(dim=0) #(640, )
        mean_anorm_prompt = torch.cat(mean_anorm_prompt, dim=0).mean(dim=0) #(640, )
        self._text_embs = torch.stack([mean_norm_prompt, mean_anorm_prompt], dim=0) #(2, 640)

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

    @property
    def text_embs(self):
        return self._text_embs
