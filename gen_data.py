import torch
import argparse

from main import Copula, sample


def gen_data(phi,
             ndims,
             N,
             seed):
    torch.set_default_tensor_type(torch.DoubleTensor)
    net = Copula(phi)

    s = sample(net, ndims, N, seed=seed)
    log_ll = -torch.log(net(s, 'pdf'))

    print('mean log_ll:', torch.mean(log_ll))

    plot_samples(s)
    return s, log_ll


def plot_samples(s):
    s_np = s.detach().numpy()
    assert s_np.ndim == 2, 'Can only plot 2d array of samples.'

    import matplotlib.pyplot as plt

    s_np = s.detach().numpy()
    plt.scatter(s_np[:, 0], s_np[:, 1])
    plt.show()
