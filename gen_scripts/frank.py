import torch
import pickle
from phi_listing import FrankPhi
from sacred import Experiment
from gen_data import gen_data, plot_samples
from sacred.observers import FileStorageObserver
ex = Experiment('LOG_DATA_frank')

torch.set_default_tensor_type(torch.DoubleTensor)


@ex.config
def cfg():
    Phi = FrankPhi
    phi_name = 'FrankPhi'
    theta = 15.
    N = 10000
    ndims = 2
    seed = 142857


@ex.capture
def get_info(_run):
    return _run._id


@ex.automain
def run(Phi, ndims, theta, N, seed):

    phi = Phi(torch.tensor(theta))
    id = get_info()
    s, log_ll = gen_data(phi, ndims, N, seed)
    print('avg_log_likelihood', torch.mean(log_ll))

    import matplotlib.pyplot as plt
    plt.scatter(s.detach().numpy()[:, 0], s.detach().numpy()[:, 1])
    plt.show()

    d = {'samples': s, 'log_ll': log_ll}
    pickle.dump(d, open('./data/frank%s.p' % id, 'wb'))
    ex.add_artifact('./data/frank%s.p' % id)


if __name__ == '__main__':
    print('Sample usage: python -m gen_scripts.frank -F gen_frank')
