import torch
import pickle
from phi_listing import ClaytonPhi
from sacred import Experiment
from gen_data import gen_data, plot_samples
from sacred.observers import FileStorageObserver
ex = Experiment('LOG_DATA_clayton')

torch.set_default_tensor_type(torch.DoubleTensor)


@ex.config
def cfg():
    Phi = ClaytonPhi
    phi_name = 'ClaytonPhi'
    theta = torch.tensor(5.)
    N = 10000
    ndims = 2
    seed = 142857


@ex.capture
def get_info(_run):
    return _run._id


@ex.automain
def run(Phi, ndims, theta, N, seed):

    phi = Phi(theta)
    id = get_info()
    s = phi.sample(ndims, N)
    log_ll = torch.log(phi.pdf(s))
    print('avg_log_likelihood', torch.mean(log_ll))

    import matplotlib.pyplot as plt
    plt.scatter(s.detach().numpy()[:, 0], s.detach().numpy()[:, 1])
    plt.show()

    d = {'samples': s, 'log_ll': log_ll}
    pickle.dump(d, open('./data/clayton%s.p' % id, 'wb'))
    ex.add_artifact('./data/clayton%s.p' % id)


if __name__ == '__main__':
    print('Sample usage: python -m gen_scripts.clayton -F gen_clayton')
