from main import Copula
from dirac_phi import DiracPhi
import pickle
from sacred import Experiment
from train import load_data, load_log_ll, expt
import torch
from sacred.observers import FileStorageObserver

identifier = 'learn_clayton'
ex = Experiment('LOG_learn_clayton')

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
        num_epochs, batch_size, chkpt_freq):
    id = get_info()
    identifier_id = '%s%s' % (identifier, id)
    train_data, test_data = load_data(data_name, num_train, num_test)
    train_ll, test_ll = load_log_ll(data_name, num_train, num_test)

    print('Computing ground truth manually because tagged log likelihood is wrong')
    from phi_listing import ClaytonPhi
    gt_phi = ClaytonPhi(torch.tensor(5.))
    cop = Copula(gt_phi)
    train_ll = -torch.log(cop(train_data, mode='pdf'))
    test_ll = -torch.log(cop(test_data, mode='pdf'))

    print('train_ll', torch.mean(train_ll))
    print('test_ll', torch.mean(test_ll))

    print('Train ideal ll:', torch.mean(train_ll))
    print('Test ideal ll:', torch.mean(test_ll))

    phi = Phi(depth, widths, lc_w_range, shift_w_range)
    net = Copula(phi)
    expt(train_data, test_data, net, optim_name,
         optim_args, identifier_id, num_epochs, batch_size, chkpt_freq)


if __name__ == '__main__':
    print('Sample usage: python -m train_scripts.clayton -F learn_clayton')
