"""
PyTorch implementation of paramatric t-SNE.
(C) Kai Xu
University of Edinburgh, 2017
"""


from __future__ import print_function

import math

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

from helper import *
from rbm import *

def preprocess_img(xs):
    """
    Process grey image matrix
    by normalizing entries to between 0 and 1.
    """

    x_num, x_w, x_h = xs.size()
    xs = xs.view(x_num, x_w * x_h)
    xs = xs.float()
    xs = xs / 255   # entries was between 0 and 255

    return xs

def pairwise(xs):
    """
    Computer the pairwise distances.

    Ref: https://lvdmaaten.github.io/tsne/
    """

    x_num, x_dim = xs.size()

    x_sum = (xs ** 2.0).sum(1).view(x_num, 1).expand(x_num, x_num)
    pws = x_sum + x_sum.t() - 2 * torch.matmul(xs, xs.t())

    return pws

def entropy(ps):
    """
    Compute the entropy of given probablity distribution.
    """

    return -(ps * torch.log(ps) * 1.4426950408889634).sum()

def perplexity(ps):
    """
    Compute the perplexity of given probablity distribution.
    """

    return 2 ** entropy(ps)

def entropy_stable(ps, pws, tau):
    """
    Stable version of entropy computation for P in the t-SNE context.
    """

    ps_sum = ps.sum()

    return math.log(ps_sum, 2) + tau * (pws * ps).sum() / ps_sum

def pairwise2gauss(pws, ppl_target, tol=1e-3):
    """
    Compute P.
    """

    # Convert target perplexity to entropy
    entropy_target = math.log(ppl_target, 2)

    # Inialize P and precisions for each point
    x_num, _ = pws.size()
    pjgi = torch.zeros(x_num, x_num).cuda()
    taus = torch.ones(x_num).cuda()

    for j in range(x_num):

        done = False    # flag to implement a do-while loop
        n_iter = 0      # maximum iteration used to find precision

        # Upper and lower bound for binary search
        tau_min, tau_max = None, None

        # Perform binary search until 
        # - the target perplexity is found, or
        # - the maximum iteration is reach
        while not done and n_iter <= 50:

            n_iter += 1

            # Fetch distances besides j
            if j == 0:
                
                pws_j = pws[1:,j]

            elif j == x_num - 1:

                pws_j = pws[:x_num,j]

            else:

                pws_j = torch.cat([pws[:j,j], pws[j+1:,j]])
            
            # Compute P
            pjgi_j = torch.exp(-pws_j * taus[j])

            # Prevent P simply being all zeros; if this
            # happens, it means precision is too large - 
            # then we decrease it by diving by 2
            while pjgi_j.sum() == 0:

                taus[j] /= 2
                pjgi_j = torch.exp(-pws_j * taus[j])
                print("[pairwise2gauss] tau is too large; set to %f" % (taus[j]))

            # Normalize P
            pjgi_j = pjgi_j / pjgi_j.sum()

            # Compute entropy and difference between target
            entropy = entropy_stable(pjgi_j, pws_j, taus[j])
            entropy_diff = entropy - entropy_target

            # Binary search for precision
            if abs(entropy_diff) <= tol:

                done = True

            elif entropy_diff < 0:

                tau_min = taus[j]

                if tau_max:

                    taus[j] = (taus[j] + tau_max) / 2
                    tau_max = taus[j]

                else:

                    taus[j] *= 2

            else:

                tau_max = taus[j]

                if tau_min:

                    taus[j] = (taus[j] + tau_min) / 2
                    tau_min = taus[j]

                else:

                    taus[j] /= 2

        # Copy row-wise P (pjgi_j) to the corresponding
        # place in the intialized matrix P (pjgi)
        if j > 0:

            pjgi[:j,j] = pjgi_j[:j]
    
        if j < x_num - 1:

            pjgi[j+1:,j] = pjgi_j[j:]

    # Computer pij using the symmetric property of conditional probablity of P.
    pigj = pjgi.t()
    pij = (pjgi + pigj) / (2 * x_num)

    return pij

def pairwise2t(pws, alpha=1):
    """
    Computer Q.
    """

    n_row, _ = pws.size()
    mask_without_diag = Variable(1 - torch.diag(torch.zeros(n_row))).cuda()

    pws_tmp = (1 + pws * mask_without_diag / alpha) ** (-(alpha + 1) / 2)
    qij = pws_tmp / pws_tmp.sum()

    return qij

def KL_stable(P, Q):
    """
    Stable implementation of KL divergence.
    """

    return torch.sum(P * torch.log((P + 1e-9) / (Q + 1e-9)))

class ptSNE:
    """
    Paramtric t-Distributed Stochastic Neighbour Embedding.
    """

    def __init__(self, encoder):

        self.encoder = encoder

    def train(self, xs, batch_size, epoch_num, lr, ppl):

        # Send data and encoder to CUDA
        xs = xs.cuda()
        self.encoder = self.encoder.cuda()

        # Initialization
        x_num, x_dim = xs.size()
        
        batches = map(lambda s: (s, s + batch_size) if s + batch_size <= x_num else (s, x_num),
            torch.arange(0, x_num, batch_size).int())
        batch_num = len(batches)

        # Compute P for each batch
        Ps_l = []

        for batch_s, batch_e in batches:
            
            xs_pairwise = pairwise(xs[batch_s:batch_e, :])
            Ps_l.append(pairwise2gauss(xs_pairwise, ppl))

        # Initialize the optimizer
        # optimizer = optim.SGD(self.encoder.parameters(), lr=lr)
        optimizer = optim.Adam(
            self.encoder.parameters(), 
            lr=lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=0)

        # Training
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

                loss = KL_stable(P, Q)

                loss.backward()

                optimizer.step()

                cost_running += loss.data[0]
                    
            print("%6d, %.7f" % (epoch, cost_running))
            log["cost"].append(cost_running)
        
        fig = make_cost_plot(log["cost"])
        fig.savefig("cost.png")

    def pre_train(self, xs, batch_size, epoch_num):
        """
        Layer-wise pre-training for the encoder using RBMs.
        """

        # Initialization
        x_num, _ = xs.size()

        batches = map(lambda s: (s, s + batch_size) if s + batch_size <= x_num else (s, x_num),
            torch.arange(0, x_num, batch_size).int())
        batch_num = len(batches)

        log_skip = int(batch_num / (3 + 1)) + 1

        # Send data to GPU
        xs = xs.cuda()

        # First RBM
        rbm_1 = RBMBer(784, 500)

        alpha = 0.5
        for epoch in range(epoch_num):

            if epoch > 5:
                alpha = 0.9

            print("#sweep, #iter, error")

            for i, (batch_s, batch_e) in enumerate(batches, 1):

                this_batch_size = batch_e - batch_s

                x = xs[batch_s:batch_e, :]

                error = rbm_1.cd(x, eta=0.1, alpha=alpha, lam=0.0002).data[0]

                if i % log_skip == 0:
                    print("%6d, %5d, %.3f" %
                        (epoch + 1, i, error / this_batch_size))

        h_1 = (rbm_1.p_h_given_v(xs) >= 0.5).float()

        # Second RBM
        rbm_2 = RBMBer(500, 500)

        alpha = 0.5
        for epoch in range(epoch_num):

            if epoch > 5:
                alpha = 0.9

            print("#sweep, #iter, error")

            for i, (batch_s, batch_e) in enumerate(batches, 1):

                this_batch_size = batch_e - batch_s

                x = h_1[batch_s:batch_e, :]

                error = rbm_2.cd(x, eta=0.1, alpha=alpha, lam=0.0002).data[0]

                if i % log_skip == 0:
                    print("%6d, %5d, %.3f" %
                        (epoch + 1, i, error / this_batch_size))

        h_2 = (rbm_2.p_h_given_v(h_1) >= 0.5).float()

        # Third RBM
        rbm_3 = RBMBer(500, 2000)

        alpha = 0.5
        for epoch in range(epoch_num):

            if epoch > 5:
                alpha = 0.9

            print("#sweep, #iter, error")

            for i, (batch_s, batch_e) in enumerate(batches, 1):

                this_batch_size = batch_e - batch_s

                x = h_2[batch_s:batch_e, :]

                error = rbm_3.cd(x, eta=0.1, alpha=alpha, lam=0.0002).data[0]

                if i % log_skip == 0:
                    print("%6d, %5d, %.3f" %
                        (epoch + 1, i, error / this_batch_size))

        h_3 = (rbm_3.p_h_given_v(h_2) >= 0.5).float()

        # Fourth RBM
        rbm_4 = RBMGaussHid(2000, 2)

        alpha = 0.5
        for epoch in range(epoch_num):

            if epoch > 5:
                alpha = 0.9

            print("#sweep, #iter, error")

            for i, (batch_s, batch_e) in enumerate(batches, 1):

                this_batch_size = batch_e - batch_s

                x = h_3[batch_s:batch_e, :]

                error = rbm_4.cd(x, eta=0.001, alpha=alpha, lam=0.0002).data[0]

                if i % log_skip == 0:
                    print("%6d, %5d, %.3f" %
                        (epoch + 1, i, error / this_batch_size))

        state_dict = self.encoder.state_dict()

        # Copy weights from RBM to encoder
        state_dict["1.weight"] = rbm_1.w
        state_dict["2.weight"] = rbm_2.w
        state_dict["3.weight"] = rbm_3.w
        state_dict["4.weight"] = rbm_4.w

    def apply(self, xs):
        
        self.encoder.eval()

        xs_2d = self.encoder(Variable(xs, volatile=True).cuda()).cpu()

        return xs_2d
