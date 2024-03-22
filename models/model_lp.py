
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
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.models import LabelPropagation


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
            self.gnn1 = GATConv(640, 64)
            self.gnn2 = GATConv(64, 64)
            self.gnn3 = GATConv(64, 640)
    def forward(self, x, edges):
        x = F.relu(self.gnn1(x, edges.T))
        x = F.relu(self.gnn2(x, edges.T))
        x = self.gnn3(x, edges.T)
        return x


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(640, 64),
            nn.ReLU(),
            nn.Linear(64, 640)
        )

    def forward(self, x):
        return self.seq(x)


class RegionClipLPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pad_green = self.config['pad']['pad_green']
        self.net_type = config['gnn']['net_type']
        self.mode = config['st']['mode']
        self.clip, _, self._transform = create_model_and_transforms(
            model_name=config['clip']['model_name'],
            pretrained=config['clip']['pretrained']
        )
        self.clip = self.clip.to('cuda')
        if self.net_type == 'gnn':
            self.teacher = GNN(config).to('cuda')
        elif self.net_type == 'linear':
            self.teacher = NN().to('cuda')
        self.student = NN().to('cuda')
        self.learned_object = Learned_Object(self.clip).cuda()

        for param in self.clip.parameters():
            param.requires_grad = False

        if self.mode == 'teacher':
            for param in self.teacher.parameters():
                param.requires_grad=True
            #for param in self.learned_object.prompt_linear.parameters():
                #param.requires_grad=True
            for param in self.student.parameters():
                param.requires_grad=False

        elif self.mode == 'student':
            for param in self.teacher.parameters():
                param.requires_grad=False
            #for param in self.learned_object.prompt_linear.parameters():
                #param.requires_grad=False
            for param in self.student.parameters():
                param.requires_grad=True
        else:
            raise Exception('plz specifiy mode')

        self.clip.eval()

    def forward(self, x, pad_green=False):
        x = x.cuda()
        batch_size = x.shape[0]
        temp = self.config['clip']['temp']
        #img level prediction
        if self.mode == 'teacher':
            # img embs
            img_embs = self.clip.encode_image(x)  # (b, d)
            text_embs_img = self.learned_object(img_embs)  # (b, 2, d)
            img_embs = F.normalize(img_embs, dim=-1)
            text_embs_img = F.normalize(text_embs_img, dim=-1)
            img_preds = (img_embs[:, None, :] * text_embs_img).sum(dim=-1) / temp  # (b, 2)
            #img_preds = img_preds.softmax(dim=-1)
        else:
            img_preds = []

        batch_region_embs = []
        batch_regions = []
        batch_edges = []
        batch_preds_t = []
        batch_nodes_t = []
        batch_preds_s = []
        batch_nodes_s = []
        batch_anomaps = []
        for i in range(batch_size):
            # region sampling
            x_i = x[i]
            regions, edges = super_pixel_graph_construct(
                x_i, **self.config['superpixel'])
            x_i = region_sampling(x_i, regions, pad_green)  # (N, 3, h, w)
            region_embs = self.clip.encode_image(x_i)  # (N, d)
            region_embs = region_embs.view(-1, region_embs.shape[-1])
            batch_region_embs.append(region_embs)  # (N, d)
            batch_regions.append(regions)
            batch_edges.append(edges)

            #forward
            if self.net_type == 'gnn':
                region_nodes_t = self.teacher(region_embs, edges)
            elif self.net_type == 'linear':
                region_nodes_t = self.teacher(region_embs)
            batch_nodes_t.append(region_nodes_t)
            if self.mode == 'teacher':
                region_max_node_t, _ = region_nodes_t.max(dim=0, keepdim=True)  # (1, d)
                region_text_node_t = text_embs_img[i][None, ...]
                region_max_node_t = F.normalize(region_max_node_t, dim=-1)
                region_text_node_t = F.normalize(region_text_node_t, dim=-1)
                node_pred = (region_max_node_t[:, None, :] * region_text_node_t).sum(dim=-1) / temp
                batch_preds_t.append(node_pred)
            elif self.mode == 'student':
                region_nodes_s = self.student(region_embs)
                anomap = anomaly_map_gen(region_nodes_s, region_nodes_t)
                batch_nodes_s.append(region_nodes_s)
                batch_anomaps.append(anomap)

        if self.mode == 'teacher':
            batch_preds_t = torch.cat(batch_preds_t)

        return img_preds, batch_preds_t, batch_preds_s,\
               batch_nodes_t, batch_nodes_s, batch_anomaps


def anomaly_map_gen(s, t):
    diff = (s - t) ** 2 #(N, d)
    return diff.mean(dim=-1, keepdim=True)






