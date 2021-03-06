from sacred import Experiment
import scipy
from sklearn.model_selection import train_test_split
import os
import numpy as np
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from main import Copula

goog_f = open('data/stocks/goog/close.vals', 'r')
goog = np.array(list(map(float, goog_f.readlines())))

fb_f = open('data/stocks/fb/close.vals', 'r')
fb = np.array(list(map(float, fb_f.readlines())))


identifier = 'goog_fb_stocks_gumbel'
ex = Experiment('goog_fb_stocks_gumbel')

torch.set_default_tensor_type(torch.DoubleTensor)


def add_train_random_noise(data, num_adds):
    new_data = np.random.rand(num_adds, data.shape[1])
    print(data.shape)
    print(new_data.shape)
    return np.concatenate((data, new_data), axis=0)


X = np.concatenate((goog[:, None], fb[:, None]), axis=1)
X_train, X_test, _, _ = train_test_split(
    X, X, shuffle=True, random_state=142857)
# X_train, X_test, _, _ = train_test_split(
#     X, X, shuffle=True, random_state=714285)
# X_train, X_test, _, _ = train_test_split(
#     X, X, shuffle=True, random_state=571428)
# X_train, X_test, _, _ = train_test_split(
#     X, X, shuffle=True, random_state=857142)
# X_train, X_test, _, _ = train_test_split(
#     X, X, shuffle=True, random_state=285714)

nfeats = X_test.shape[1]

# Normalize data.
for z in [X_train, X_test]:
    ndata = z.shape[0]
    gap = 1./(ndata+1)
    for i in range(nfeats):
        z[:, i] = scipy.stats.rankdata(z[:, i], 'ordinal')*gap

w = X.copy()
ndata = w.shape[0]
gap = 1./(ndata+1)
for i in range(nfeats):
    w[:, i] = scipy.stats.rankdata(w[:, i], 'ordinal')*gap
# np.savetxt('stocks_transformed.points', w)


def get_optim(name, net, args):
    if name == 'SGD':
        optimizer = optim.SGD(net.parameters(), args['lr'], args['momentum'])
    else:
        assert False

    return optimizer


@ex.config
def cfg():
    x_index = 0
    y_index = 1

    x_flip, y_flip = False, False

    optim_name = 'SGD'
    optim_args = \
        {
            'lr': 1e-5,
            'momentum': 0.9
        }
    num_epochs = 10000000
    # batch_size = 1000
    batch_size = 200
    chkpt_freq = 500

    from phi_listing import GumbelPhi
    Phi = GumbelPhi
    phi_name = 'GumbelPhi'

    # Initial parameters.
    initial_theta = 5.0

    frac_rand = 0.01

@ex.capture
def get_info(_run):
    return _run._id


def expt(train_data, val_data,
         net,
         optim_name,
         optim_args,
         identifier,
         num_epochs=1000,
         batch_size=100,
         chkpt_freq=50,
         ):

    os.mkdir('./checkpoints/%s' % identifier)
    os.mkdir('./sample_figs/%s' % identifier)
    # os.mkdir('./psi_figs/%s' % identifier)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_data, batch_size=1000000, shuffle=True)

    optimizer = get_optim(optim_name, net, optim_args)

    train_loss_per_epoch = []

    for epoch in range(num_epochs):
        loss_per_minibatch = []
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()

            d = torch.tensor(data, requires_grad=True)
            p = net(d, mode='pdf')

            logloss = -torch.sum(torch.log(p))
            reg_loss = logloss
            reg_loss.backward()
            scalar_loss = (reg_loss/p.numel()).detach().numpy().item()

            loss_per_minibatch.append(scalar_loss)
            optimizer.step()

        train_loss_per_epoch.append(np.mean(loss_per_minibatch))
        print('Training loss at epoch %s: %s' %
              (epoch, train_loss_per_epoch[-1]))

        if epoch % chkpt_freq == 0:
            print('Checkpointing')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': logloss,
            }, './checkpoints/%s/epoch%s' % (identifier, epoch))

            print('Evaluating validation loss')
            for j, val_data in enumerate(val_loader, 0):
                net.zero_grad()
                val_p = net(val_data, mode='pdf')
                val_loss = -torch.mean(torch.log(val_p))
            print('Average validation loss %s' % val_loss)


@ex.automain
def run(x_index, y_index,
        x_flip, y_flip,
        Phi,
        initial_theta,
        optim_name, optim_args,
        num_epochs, batch_size, chkpt_freq,
        frac_rand):
    id = get_info()
    identifier_id = '%s%s' % (identifier, id)

    train_data = X_train[:, [x_index, y_index]]
    train_data = add_train_random_noise(train_data, int(train_data.shape[0]*frac_rand))
    test_data = X_test[:, [x_index, y_index]]

    if x_flip:
        train_data[:, 0] = 1-train_data[:, 0]
        test_data[:, 0] = 1-test_data[:, 0]

    if y_flip:
        train_data[:, 1] = 1-train_data[:, 1]
        test_data[:, 1] = 1-test_data[:, 1]

    phi = Phi(torch.tensor(initial_theta))
    net = Copula(phi)
    expt(train_data, test_data, net, optim_name,
         optim_args, identifier_id, num_epochs, batch_size, chkpt_freq)


if __name__ == '__main__':
    print('Sample usage: python -m train_scripts.stocks.train_with_gumbel -F learn_stocks_with_gumbel')
