import numpy as np
from skimage.segmentation import slic
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Resize, RandomErasing, Lambda, Normalize, ColorJitter, Grayscale

def super_pixel_graph_construct(img, numSegments = 100, sigma = 3):
    '''
    input: image with size (h, w, 3)
    output: super-pixel segments and edges.
    Edges are consturcted by sliding 3x3 kernel to determine whether two regions are connected.
    If a pixel's neighbor is from other region, then two regions are connected.
    '''
    img = img.permute(1, 2, 0).cpu()
    segments = slic(img, n_segments = numSegments, sigma = sigma)
    regions = segments-1
    h, w = regions.shape
    _edges = []
    regions_pad = np.pad(regions,[(2,2),(2,2)],"constant",constant_values=(-1))
    for i in range(2, h+2):
        for j in range(2, w+2):
            region = regions_pad[i, j]
            filter = regions_pad[i-2:i+3,j-2:j+3].flatten()
            filter = filter[[6,7,8,11,13,16,17,18]]
            for tmp in filter:
                if tmp != region and tmp!=-1:
                    if [region, tmp] not in _edges:
                        _edges.append([region, tmp])
    _edges = np.array(_edges)
    edges = []
    #re-ordering based on regions
    for region_idx in np.unique(regions):
        region_edges = _edges[_edges[:, 0]==region_idx]
        edges.append(region_edges)
    edges = np.concatenate(edges,axis=0)
    regions, edges = torch.tensor(regions), torch.tensor(edges, dtype = torch.int64)
    regions, edges = regions.cuda(), edges.cuda()
    return regions, edges

"""
def region_sampling(x, regions):
    '''
    :param x: (3, h, w)
    :param regions: (h, w)
    :return output: (N, 3, h, w)
    '''
    num_regions = len(regions.unique())
    output = []
    #filtering based on region order
    for i in range(num_regions):
        xi = x.clone()
        xi[:, regions!=i] = 0.
        output.append(xi)
    output = torch.stack(output, dim=0)
    return output
"""


def region_sampling(x, regions, pad_green=False):
    '''
    :param x: (3, h, w)
    :param regions: (h, w)
    :return output: (N, 3, h, w)
    '''
    _, h, w = x.shape
    grid_x, grid_y = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=0).cuda()
    resize = Resize([h, w])
    num_regions = len(regions.unique())
    output = []
    for i in range(num_regions):
        xi = x.clone()
        xi[:, regions !=i] = 0.

        if pad_green:
            # green background
            xi[0, regions != i] = -1.7923
            xi[1, regions != i] = 2.0749
            xi[2, regions != i] = -1.4802

        #sample
        region_grid = grid[:, regions==i]
        min_ptx = region_grid[0].min()
        min_pty = region_grid[1].min()
        max_ptx = region_grid[0].max()
        max_pty = region_grid[1].max()
        xi = xi[:, min_ptx:max_ptx, min_pty:max_pty]
        xi = resize(xi)
        output.append(xi)
    output = torch.stack(output, dim=0)
    return output

mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).cuda()
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def green_object_sample():
    random_augment = Lambda(
        lambda x: torch.stack(
            [RandomErasing(
                p=1,
                value=[-1.7923,
                       torch.distributions.uniform.Uniform(0.5, 1.).sample().item(),
                       -1.4802
                       ],
                scale=(0.01, 0.05),
                ratio=(0.3, 3.3)
            )(x_) \
             for x_ in x]))
    return random_augment

def get_color_distortion(p=0.5, s=1.0):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=p)
    rnd_gray = transforms.RandomGrayscale(p=p)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    random_augment = Lambda(
        lambda x: torch.stack(
            [color_distort(x_) for x_ in x]))
    return random_augment
def black_object_sample():
    random_augment = Lambda(
        lambda x: torch.stack(
            [RandomErasing(
                p=1,
                value=[-2.1179, -1.8078, -1.8044],
                scale=(0.01, 0.05),
                ratio=(0.3, 3.3)
            )(x_) \
             for x_ in x]))
    return random_augment

def strong_augment(prob):
    random_augment = Lambda(
        lambda x: torch.stack(
            [RandomErasing(p=prob, value=torch.rand(1)[0].item())(x_) for x_ in x]))
    return random_augment


def region_augment(regions, pad_green, prob, augment_type='strong'):
    N = regions.shape[0]
    idx = np.random.choice(N, N//2, replace = False)
    aug_regions = regions[idx] * std[None, :, None, None] +\
                  mean[None, :, None, None] #(N//2, 3, h, w)
    if augment_type == 'weak':
        if pad_green:
            random_augment = black_object_sample()
        else:
            random_augment = green_object_sample()
    elif augment_type == 'strong':
        random_augment = strong_augment(prob)

    elif augment_type == 'color':
        random_augment = get_color_distortion(prob)
    else:
        raise  Exception('specify augment_type')
    aug_regions = random_augment(aug_regions)
    aug_regions = normalize(aug_regions)
    regions[idx] = aug_regions
    return regions, idx
