"""
PyTorch implementation of paramatric t-SNE.
(C) Kai Xu
University of Edinburgh, 2017
"""


from __future__ import print_function

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

from helper import *

def preprocess_img(xs):

    x_num, x_w, x_h = xs.size()
    xs = xs.view(x_num, x_w * x_h)
    xs = xs.float()
    xs = xs / 255
    # xs = (xs - 0.5) * 2

    return xs

def pairwise(xs):

    x_num, x_dim = xs.size()
    x1 = xs.unsqueeze(0).expand(x_num, x_num, x_dim)
    x2 = xs.unsqueeze(1).expand(x_num, x_num, x_dim)
    pws = ((x1 - x2) ** 2.0).sum(2)

    return pws

def pairwise2gauss(pws, ppl=1):

    x_num, _ = pws.size()
    pws_tmp = torch.exp(-pws / (2 * ppl))

    pjgi = pws_tmp / pws_tmp.sum(0).view(x_num, 1).repeat(1, x_num)
    pigj = pws_tmp / pws_tmp.sum(1).view(1, x_num).repeat(x_num, 1)

    pij = (pjgi + pigj) / (2 * x_num)

    return pij

def pairwise2t(pws, alpha=1):

    pws_tmp = (1 + pws / alpha) ** (-(alpha + 1) / 2)

    qij = pws_tmp / pws_tmp.sum()

    return qij

class ptSNE:

    def __init__(self, encoder):

        self.encoder = encoder

    def KL_without_diag(self, P, Q):

        n_row, _ = P.size()
        mask_without_diag = Variable(1 - torch.diag(torch.zeros(n_row))).cuda()
        kl = torch.sum(P * torch.log(P / Q) * mask_without_diag)
        
        return kl

    def train(self, xs, batch_size, epoch_num, lr, ppl):

        xs = xs.cuda()
        self.encoder = self.encoder.cuda()

        x_num, x_dim = xs.size()
        
        batches = map(lambda s: (s, s + batch_size) if s + batch_size <= x_num else x_num,
            torch.arange(0, x_num, batch_size).int())
        batch_num = len(batches)

        Ps_l = []

        for batch_s, batch_e in batches:
            
            xs_pairwise = pairwise(xs[batch_s:batch_e, :])
            Ps_l.append(pairwise2gauss(xs_pairwise, ppl=ppl))

        # optimizer = optim.SGD(self.encoder.parameters(), lr=lr)
        optimizer = optim.Adam(
            self.encoder.parameters(), 
            lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        # criterion = torch.nn.KLDivLoss(size_average=False)

        log = dict()
        log["cost"] = []

        for epoch in range(1, epoch_num + 1):

            print("#epoch, cost")

            cost_running = .0

            for i, (batch_s, batch_e) in enumerate(batches, 0):

                x = Variable(xs[batch_s:batch_e, :])
                P = Variable(Ps_l[i])

                optimizer.zero_grad()

                y = self.encoder(x)
                y_pairwise = pairwise(y)

                Q = pairwise2t(y_pairwise)

                # loss = criterion(torch.log(Q), P)
                loss = self.KL_without_diag(P, Q)
                
                loss.backward()

                optimizer.step()

                cost_running += loss.data[0]
                    
            print("%6d, %.7f" % (epoch, cost_running))
            log["cost"].append(cost_running)
        
        fig = make_cost_plot(log["cost"])
        fig.savefig("cost.png")

    def apply(self, xs):
        
        self.encoder.eval()

        xs_2d = self.encoder(Variable(xs, volatile=True).cuda()).cpu()

        return xs_2d
