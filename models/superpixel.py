import numpy as np
from skimage.segmentation import slic
import torch

def super_pixel_graph_construct(img, numSegments = 100, sigma = 3):
    '''
    input: image with size (h, w, 3)
    output: super-pixel segments and edges.
    Edges are consturcted by sliding 3x3 kernel to determine whether two regions are connected.
    If a pixel's neighbor is from other region, then two regions are connected.
    '''
    img = img.permute(1, 2, 0)
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
    regions, edges = torch.tensor(regions), torch.tensor(edges)
    regions, edges = regions.to(img.device), edges.to(img.device)
    return regions, edges

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

