
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
#from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.models import GAT, GCN

class Teacher_GNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        gnn_type = config['gnn']['gnn_type']
        self.heads = config['gnn']['heads']
        if gnn_type == 'GCN':
            self.gnn = GCN(896, 16, 4, 640, act = 'relu', jk='lstm', norm='LayerNorm')

        elif gnn_type == 'GAT':
            self.gnn = GAT(896, 16, 4, 640, act = 'relu', jk='lstm', norm='LayerNorm')

    def forward(self, x, edges, batch):
        return self.gnn(x, edges.T, batch = batch.long())


class Teacher_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(896, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 640)
        )

    def forward(self, x):
        return self.seq(x)


class Student_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(896, 64),
            nn.ReLU(),
            nn.Linear(64, 640)
        )
    def forward(self, x):
        return self.seq(x)


class AnomalyMapGenerator(nn.Module):
    def __init__(self):
        super().__init__()
    def compute_layer_map(self, ft, fs, img_size):
        ft = F.normalize(ft, dim=1)
        fs = F.normalize(fs, dim=1)
        layer_map = 0.5 * torch.norm(ft - fs, p=2, dim=-3, keepdim=True) ** 2
        return F.interpolate(layer_map,
                             size=img_size,
                             align_corners=False,
                             mode="bilinear",
                             )
    def forward(self, ft, fs, img_size):
        b, _, h, w = img_size
        #anomaly_map = torch.ones(b, 1, h, w).cuda()
        anomaly_map = self.compute_layer_map(ft, fs, (h, w))
        return anomaly_map


class RegionClipViTModel(nn.Module):
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
        self.clip.visual.output_tokens = True
        if self.net_type == 'gnn':
            self.teacher = Teacher_GNN(config).to('cuda')
        elif self.net_type == 'linear':
            self.teacher = Teacher_NN().to('cuda')

        self.student = Student_NN().cuda()
        self.learned_object = Learned_Object(self.clip).cuda()
        self.anomap = AnomalyMapGenerator()

        for param in self.clip.parameters():
            param.requires_grad = False

        if self.mode == 'teacher':
            for param in self.teacher.parameters():
                param.requires_grad=True
            for param in self.learned_object.prompt_linear.parameters():
                param.requires_grad=True
            for param in self.student.parameters():
                param.requires_grad=False

        elif self.mode == 'student':
            for param in self.teacher.parameters():
                param.requires_grad=False
            for param in self.learned_object.prompt_linear.parameters():
                param.requires_grad=False
            for param in self.student.parameters():
                param.requires_grad=True
        else:
            raise Exception('plz specifiy mode')

        self.clip.eval()

    def forward(self, x):
        x = x.cuda()
        batch_size = x.shape[0]
        temp = self.config['clip']['temp']
        img_embs, patch_embs = self.clip.encode_image(x)  # (b, d), (b, 225, 896)
        text_embs_img = self.learned_object(img_embs)  # (b, 2, d)
        img_embs = F.normalize(img_embs, dim=-1)
        text_embs_img = F.normalize(text_embs_img, dim=-1) # (b, 2, d)

        #img level prediction
        if self.mode == 'teacher':
            img_preds = (img_embs[:, None, :] * text_embs_img).sum(dim=-1) / temp  # (b, 2)
        else:
            img_preds = []

        patch_embs_ = patch_embs.reshape(-1, 896) #(b*225, 896)
        batch_idx = torch.cat([torch.ones((225, ))* i for i in range(batch_size)]).cuda()
        edges = torch.cat([create_edges(15) for _ in range(batch_size)]).cuda()
        nodes_t = self.teacher(patch_embs_, edges, batch_idx)
        nodes_t = nodes_t.view(batch_size, -1, 640) #(b, 225, 640)

        if self.mode == 'teacher':
            max_node_t, _ = nodes_t.max(dim=1, keepdim=True) #(b, 1, 640)
            max_node_t = F.normalize(max_node_t, dim=-1)
            node_preds_t = (text_embs_img * max_node_t).sum(dim=-1)/temp
        else:
            node_preds_t = []

        if self.mode == 'student':
            nodes_s = self.student(patch_embs) # (b, 225, 640)
            nodes_t = nodes_t.permute(0, 2, 1)
            nodes_s = nodes_s.permute(0, 2, 1)
            nodes_t = nodes_t.view(batch_size, -1, 15, 15)
            nodes_s = nodes_s.view(batch_size, -1, 15, 15)
            anomap = self.anomap(nodes_s, nodes_t, x.shape)
        else:
            nodes_s = []
            anomap = []

        return img_preds, node_preds_t, nodes_t, nodes_s, anomap



def anomaly_map_gen(s, t):
    diff = (s - t) ** 2 #(b, N, d)
    return diff.mean(dim=-1)





