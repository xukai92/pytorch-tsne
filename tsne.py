"""
PyTorch implementation of t-SNE and paramatric t-SNE.
(C) Kai Xu
University of Edinburgh, 2017
"""


from __future__ import print_function
import torch
import torch.nn.functional as F


def preprocess_img(xs):

    x_num, x_w, x_h = xs.size()
    xs = xs.view(x_num, x_w * x_h)
    xs = xs.float() / 255
    xs = (xs - 0.5) * 2

    return xs


def pairwise(xs):

    x_num, x_dim = xs.size()
    x1 = xs.unsqueeze(0).expand(x_num, x_num, x_dim)
    x2 = xs.unsqueeze(1).expand(x_num, x_num, x_dim)
    dkl2 = ((x1 - x2) ** 2.0).sum(2).squeeze()

    return dkl2
