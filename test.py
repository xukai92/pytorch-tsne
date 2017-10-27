"""
Test script for t-SNE in PyTorch.
"""


from __future__ import print_function
from tsne import *

import torch


print("[tsne.test] testing starts!")

print("[tsne.test] testing preprocess_img...")

imgs, _ = torch.load("test.pt") # 200 images from MNIST
xs = preprocess_img(imgs)

print("[tsne.test] done")

print("[tsne.test] testing pairwise...")

xs_pairwise = pairwise(xs)

print("[tsne.test] done")

print("[tsne.test] all tests pass.")
