"""
Test script for t-SNE in PyTorch.
"""


from __future__ import print_function
from tsne import *

import torch
import torch.nn as nn

print("[tsne.test] testing starts!")

print("[tsne.test] testing preprocess_img...")

imgs, labels = torch.load("test.pt") # 200 images from MNIST
xs = preprocess_img(imgs)
xs = xs[:10000,:]

print("[tsne.test] done")

print("[tsne.test] testing entroypy...")

ps = torch.Tensor([0.5, 0.5])
H = entropy(ps)
assert (H == 1)

print("[tsne.test] done")

print("[tsne.test] testing preplexity...")

assert (perplexity(ps) == 2)

print("[tsne.test] done")

print("[tsne.test] testing pairwise...")

t = torch.Tensor([[-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]).cuda()
t_pairwise = pairwise(t)
assert (t_pairwise.cpu() == torch.Tensor([[0.0, 4.0, 9.0], [4.0, 0.0, 1.0], [9.0, 1.0, 0.0]])).sum() == 9

print("[tsne.test] done")

print("[tsne.test] testing pairwise2gauss...")

Ps = pairwise2gauss(t_pairwise, 30)
# assert (Ps.cpu() - torch.Tensor([[0.0, 0.184, 0.016], [0.184, 0.0, 0.300], [0.016, 0.300, 0.0]])).sum() < 1e-3

print("[tsne.test] done")

print("[tsne.test] testing pairwise2t...")

Qs = pairwise2t(t_pairwise)

print("[tsne.test] done")

exit()

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

ptsne.pre_train(xs, 100, 50)
ptsne.train(xs, 100, 30, 1e-3, 30)

xs_2d = ptsne.apply(xs)

fig = make_2d_plot(xs_2d, labels)
fig.savefig("vis_scatter.png")

print("[tsne.test] testing pairwise...")

print("[tsne.test] all tests pass.")
