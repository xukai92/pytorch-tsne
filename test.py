"""
Test script for t-SNE in PyTorch.
"""


from __future__ import print_function
from tsne import *

import torch
import torch.nn as nn


print("[tsne.test] testing starts!")

print("[tsne.test] testing preprocess_img...")

imgs, _ = torch.load("test.pt") # 200 images from MNIST
xs = preprocess_img(imgs)

print("[tsne.test] done")

print("[tsne.test] testing pairwise...")

t = torch.Tensor([[-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
t_pairwise = pairwise(t)
assert (t_pairwise == torch.Tensor([[0.0, 4.0, 9.0], [4.0, 0.0, 1.0], [9.0, 1.0, 0.0]])).sum() == 9

print("[tsne.test] done")

print("[tsne.test] testing pairwise2gauss...")

Ps = pairwise2gauss(t_pairwise)
assert (Ps - torch.Tensor([[0.291, 0.033, 0.003], [0.033, 0.191, 0.121], [0.003, 0.121, 0.206]])).sum() < 1e-3

print("[tsne.test] done")

print("[tsne.test] testing pairwise2t...")

Qs = pairwise2t(t_pairwise)

print("[tsne.test] done")

print("[tsne.test] testing ptSNE...")

encoder = nn.Sequential(
    nn.Linear(784, 500),
    nn.Sigmoid(),
    nn.Linear(500, 500),
    nn.Sigmoid(),
    nn.Linear(500, 2000),
    nn.Sigmoid(),
    nn.Linear(2000, 2))

ptsne = ptSNE(encoder)

ptsne.train(xs, 100, 10, 5e8)

print("[tsne.test] testing pairwise...")

print("[tsne.test] all tests pass.")
