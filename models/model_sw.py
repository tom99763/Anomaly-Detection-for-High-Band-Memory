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
            self.bottleneck = GCNConv(640, 64)
            #strong
            self.gnn_s1 = GCNConv(64, 64)
            self.gnn_s2 = GCNConv(64, 640)
            #weak
            self.gnn_w1 = GCNConv(64, 64)
            self.gnn_w2 = GCNConv(64, 640)

        elif gnn_type == 'GAT':
            self.bottleneck = GATConv(640, 64, heads = self.heads, concat=False)
            #strong
            self.gnn_s1 = GATConv(64, 64, heads = self.heads, concat=False)
            self.gnn_s2 = GATConv(64, 640, heads = self.heads, concat=False)
            #weak
            self.gnn_w1 = GATConv(64, 64, heads=self.heads, concat=False)
            self.gnn_w2 = GATConv(64, 640, heads=self.heads, concat=False)
    def forward(self, x, edges):
        xb = F.relu(self.bottleneck(x, edges.T))
        #strong
        xs = F.relu(self.gnn_s1(xb, edges.T))
        xs = F.relu(self.gnn_s2(xs, edges.T))
        #weak
        xw = F.relu(self.gnn_w1(xb, edges.T))
        xw = F.relu(self.gnn_w2(xw, edges.T))
        return xs, xw

class RegionClipSWModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.level = config['gnn']['level']
        self.linear_probe = config['prompt']['linear_probe']
        self.pad_green = self.config['pad']['pad_green']
        self.net_type = config['gnn']['net_type']
        self.clip, _, self._transform = create_model_and_transforms(
            model_name=config['clip']['model_name'],
            pretrained=config['clip']['pretrained']
        )
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip = self.clip.to('cuda')
        self.gnn = GNN(config).to('cuda')

        #self.prompt = Learned_Prompt(config, self.clip).to('cuda')
        self.get_text_embs()
        self.temp = self.config['clip']['temp']

    def forward(self, x, pad_green=False, augment = False):
        x = x.cuda()

        #strong
        batch_region_embs_s, batch_edges_s, \
        batch_regions_s, batch_anorm_idx_s = self.get_region_embs(x, pad_green, augment, 'strong')

        #weak
        batch_region_embs_w, batch_edges_w, \
        batch_regions_w, batch_anorm_idx_w = self.get_region_embs(x, pad_green, augment, 'weak')

        #text emb
        text_embs = F.normalize(self.text_embs).to(x.device) #(2, 640)

        #strong pred
        batch_region_node_preds_ss, batch_region_node_preds_sw = self.node_predict(
            batch_region_embs_s, batch_edges_s, text_embs
        )
        #weak pred
        batch_region_node_preds_ws, batch_region_node_preds_ww = self.node_predict(
            batch_region_embs_w, batch_edges_w, text_embs
        )
        s_pred = (batch_region_node_preds_ss, batch_region_node_preds_sw, batch_anorm_idx_s)
        w_pred = (batch_region_node_preds_ws, batch_region_node_preds_ww, batch_anorm_idx_w)
        return s_pred, w_pred

    def node_predict(self, batch_region_embs, batch_edges, text_embs):
        batch_size = len(batch_region_embs)
        batch_region_node_preds_s = []
        batch_region_node_preds_w = []
        for i in range(batch_size):
            region_embs = batch_region_embs[i].cuda()  # (N, d)
            edges = batch_edges[i].cuda()
            region_nodes_s, region_nodes_w = self.gnn(region_embs, edges)
            # node prediction
            region_nodes_s = F.normalize(region_nodes_s, dim=1)
            node_pred_s = region_nodes_s @ text_embs.T / self.temp  # (N, 2)
            region_nodes_w = F.normalize(region_nodes_w, dim=1)
            node_pred_w = region_nodes_w @ text_embs.T / self.temp  # (N, 2)
            batch_region_node_preds_s.append(node_pred_s)
            batch_region_node_preds_w.append(node_pred_w)
        return batch_region_node_preds_s, batch_region_node_preds_w

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

    def get_region_embs(self, x, pad_green=False, augment=False, augment_type=None):
        batch_size = x.shape[0]
        batch_region_embs = []
        batch_regions = []
        batch_edges = []
        batch_anorm_idx = []
        for i in range(batch_size):
            # regions: (h, w), edges:[...]
            x_i = x[i]
            regions, edges = super_pixel_graph_construct(
                x_i, **self.config['superpixel'])
            x_i = region_sampling(x_i, regions, pad_green)  # (N, 3, h, w)
            if augment:
                x_i, idx = region_augment(x_i, self.pad_green, augment_type)
                batch_anorm_idx.append(idx)
            region_embs = self.clip.encode_image(x_i)  # (N, d)
            region_embs = region_embs.view(-1, region_embs.shape[-1])
            batch_region_embs.append(region_embs)  # (N, d)
            batch_regions.append(regions)
            batch_edges.append(edges)
        return batch_region_embs, batch_edges, batch_regions, batch_anorm_idx

    @property
    def text_embs(self):
        return self._text_embs
