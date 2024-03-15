
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
from .hand_craft_prompt import *
from torch_geometric.nn import GCNConv, GATConv
from open_clip.tokenizer import tokenize
from torch.autograd import Variable

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
    def __init__(self):
        super().__init__()
        self.attn = NN()
        self.drop_ratio = 0.05
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


class RegionClipCTModel(nn.Module):
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
        if self.net_type == 'gnn':
            self.gnn = GNN(config).to('cuda')
        elif self.net_type == 'linear':
            self.nn = NN().to('cuda')

        #self.noise_inject = NoiseInjection()
        self.noise_inject = DimMasking()
        self.get_text_embs()

    def forward(self, x, pad_green=False, generate_unlabeled= False, augment=False):
        if augment:
            for params in self.gnn.parameters():
                params.requires_grad=False
            for param in self.noise_inject.parameters():
                param.requires_grad=True
        else:
            for params in self.gnn.parameters():
                params.requires_grad=True
            for param in self.noise_inject.parameters():
                param.requires_grad=False

        x = x.cuda()
        batch_size = x.shape[0]
        batch_region_embs, batch_edges, \
        batch_regions, batch_unlabeled_idx = self.get_region_embs(x, pad_green, generate_unlabeled)

        text_embs = F.normalize(self.text_embs).to(x.device)  # (2, 640)
        temp = self.config['clip']['temp']

        #forward
        batch_region_node_preds = []
        batch_region_node_preds_aug = []
        batch_region_nodes = []
        batch_region_nodes_aug = []

        for i in range(batch_size):
            #no augment
            region_embs = batch_region_embs[i].to(x.device)  # (N, d)
            edges = batch_edges[i].to(x.device)
            region_nodes = self.gnn(region_embs, edges)
            batch_region_nodes.append(region_nodes)

            #pred
            region_nodes = F.normalize(region_nodes, dim=1)
            node_pred = region_nodes @ text_embs.T / temp  # (N, 2)
            batch_region_node_preds.append(node_pred)

            #augment
            if generate_unlabeled:
                idx = batch_unlabeled_idx[i]
                region_embs_aug = region_embs.clone()
                region_embs_aug[idx] = self.noise_inject(region_embs_aug[idx])
                region_nodes_aug = self.gnn(region_embs_aug, edges)
                batch_region_nodes_aug.append(region_nodes_aug)

                #pred
                region_nodes_aug = F.normalize(region_nodes_aug, dim=1)
                node_pred_aug = region_nodes_aug @ text_embs.T / temp  # (N, 2)
                batch_region_node_preds_aug.append(node_pred_aug)

        return batch_region_node_preds, batch_region_nodes,\
               batch_region_node_preds_aug, batch_region_nodes_aug, batch_unlabeled_idx


    def get_text_embs(self):
        #normal_embs, anormal_embs = self.prompt()
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

    def get_region_embs(self, x, pad_green=False, generate_unlabeled=False):
        batch_size = x.shape[0]
        batch_region_embs = []
        batch_regions = []
        batch_edges = []
        batch_unlabeled_idx = []
        for i in range(batch_size):
            # regions: (h, w), edges:[...]
            x_i = x[i]
            regions, edges = super_pixel_graph_construct(
                x_i, **self.config['superpixel'])
            x_i = region_sampling(x_i, regions, pad_green)  # (N, 3, h, w)
            if generate_unlabeled:
                x_i, idx = region_augment(x_i, self.pad_green, 'color')
                batch_unlabeled_idx.append(idx)
            region_embs = self.clip.encode_image(x_i)  # (N, d)
            region_embs = region_embs.view(-1, region_embs.shape[-1])
            batch_region_embs.append(region_embs)  # (N, d)
            batch_regions.append(regions)
            batch_edges.append(edges)
        return batch_region_embs, batch_edges, batch_regions, batch_unlabeled_idx

    @property
    def text_embs(self):
        return self._text_embs





