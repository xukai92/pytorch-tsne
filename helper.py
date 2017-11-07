import numpy as np

import matplotlib
matplotlib.use('Agg')           # for disabling graphical UI
import matplotlib.pyplot as plt
plt.style.use('ggplot')         # for better looking
import matplotlib.cm as cm      # for generating color list
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def make_cost_plot(cost_his,
                   title="Loss v.s. #epochr",
                   xlabel="#epoch",
                   ylabel="loss",
                   start=0):
    """
    Make a line plot of loss history.
    """

    fig = plt.figure(figsize=(12, 6.75))

    p, = plt.plot(range(start, len(cost_his)), cost_his[start:])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([p], [xlabel])

    return fig

def make_2d_plot(xs_2d, labels):

    xs_2d = xs_2d.data.numpy()

    dim_1_min = min(xs_2d[:, 0])
    dim_1_max = max(xs_2d[:, 0])
    dim_2_min = min(xs_2d[:, 1])
    dim_2_max = max(xs_2d[:, 1])

    colors = cm.rainbow(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(16, 9))

    plt.scatter(xs_2d[:, 0], xs_2d[:, 1], alpha=0.7,
                color=map(lambda i: colors[i], labels.numpy()))
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.title("Low dimensional representation of MNIST data")
    
    ax.set_xlim(dim_1_min, dim_1_max)
    ax.set_ylim(dim_2_min, dim_2_max)

    return fig
