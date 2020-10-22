from main import Copula
from dirac_phi import DiracPhi
import pickle
from sacred import Experiment
from train import load_data, load_log_ll, expt_cdf_noisy
import torch
from sacred.observers import FileStorageObserver

identifier = 'learn_clayton_noisy'
ex = Experiment('LOG_learn_clayton_noisy')

torch.set_default_tensor_type(torch.DoubleTensor)


@ex.config
def cfg():
    data_name = './data/clayton1.p'

    num_train, num_test = 2000, 1000
    optim_name = 'SGD'
    optim_args = \
        {
            'lr': 1e-5,
            'momentum': 0.9
        }
    num_epochs = 10000000
    batch_size = 200
    chkpt_freq = 50

    seed_noise = 142857
    width_noise = 0.5

    Phi = DiracPhi
    phi_name = 'DiracPhi'

    # Initial parameters.
    depth = 2
    widths = [10, 10]
    lc_w_range = (0, 1.0)
    shift_w_range = (0., 2.0)


@ex.capture
def get_info(_run):
    return _run._id


@ex.automain
def run(data_name, num_train, num_test, Phi,
        depth, widths, lc_w_range, shift_w_range,
        optim_name, optim_args,
        width_noise, seed_noise,
        num_epochs, batch_size, chkpt_freq):
    id = get_info()
    identifier_id = '%s%s' % (identifier, id)
    train_data, test_data = load_data(data_name, num_train, num_test)
    train_ll, test_ll = load_log_ll(data_name, num_train, num_test)

    print('Train ideal ll:', torch.mean(train_ll))
    print('Test ideal ll:', torch.mean(test_ll))

    phi = Phi(depth, widths, lc_w_range, shift_w_range)
    net = Copula(phi)
    expt_cdf_noisy(train_data, test_data, net, optim_name,
                   optim_args, identifier_id,
                   width_noise, seed_noise,
                   num_epochs, batch_size, chkpt_freq)


if __name__ == '__main__':
    print('Sample usage: python -m train_scripts.clayton_noisy -F learn_clayton_noisy')
