from main import Copula
from torch.autograd import Function, gradcheck
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from main import sample
from dirac_phi import DiracPhi

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import scipy

from sacred import Experiment

identifier = 'boston_housing'
ex = Experiment('boston_housing')

torch.set_default_tensor_type(torch.DoubleTensor)


def add_train_random_noise(data, num_adds):
    new_data = np.random.rand(num_adds, data.shape[1])
    return np.concatenate((data, new_data), axis=0)


X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=142857)
X_train = np.concatenate((X_train, y_train[:, None]), axis=1)
X_test = np.concatenate((X_test, y_test[:, None]), axis=1)

nfeats = X_test.shape[1]

# Normalize data based on ordinal rankings.
for z in [X_train, X_test]:
    ndata = z.shape[0]
    gap = 1./(ndata+1)
    for i in range(nfeats):
        z[:, i] = scipy.stats.rankdata(z[:, i], 'ordinal')*gap

# Potentially inject noise into data: comment if you do not want noise.
X_train = add_train_random_noise(X_train, int(X_train.shape[0]*0.01))

"""
FEATURE DESCRIPTIONS. We are interested in features 0 and -1 (price)

0. crim
per capita crime rate by town.

1. zn
proportion of residential land zoned for lots over 25,000 sq.ft.

2. indus
proportion of non-retail business acres per town.

3. chas
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

4. nox
nitrogen oxides concentration (parts per 10 million).

5. rm
average number of rooms per dwelling.

6. age
proportion of owner-occupied units built prior to 1940.

7. dis
weighted mean of distances to five Boston employment centres.

8. rad
index of accessibility to radial highways.

9. tax
full-value property-tax rate per \$10,000.

10. ptratio
pupil-teacher ratio by town.

11. black
1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

12. lstat
lower status of the population (percent).

13 [y] . medv
median value of owner-occupied homes in \$1000s.

"""


def get_optim(name, net, args):
    if name == 'SGD':
        optimizer = optim.SGD(net.parameters(), args['lr'], args['momentum'])
    elif name == 'Adam':
        # TODO: add in more.
        optimizer = optim.Adam(net.parameters(), args['lr'])
    elif name == 'RMSprop':
        # TODO: add in more.
        optimizer = optim.RMSprop(net.parameters(), args['lr'])

    return optimizer


@ex.config
def cfg():
    x_index = 0
    y_index = 13

    # Flip around the line y = 0.5 to make negative correlations positive.
    x_flip, y_flip = True, False

    optim_name = 'SGD'
    optim_args = \
        {
            'lr': 1e-5,
            'momentum': 0.9
        }
    num_epochs = 10000000
    batch_size = 200
    chkpt_freq = 500

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

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_data, batch_size=1000000, shuffle=True)

    # IMPORTANT: for this experiment, we did *not* perform hyperparameter tuning.
    # Hence, the `validation loss' here is essentially `test` loss.
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
        depth, widths, lc_w_range, shift_w_range,
        optim_name, optim_args,
        num_epochs, batch_size, chkpt_freq):
    id = get_info()
    identifier_id = '%s%s' % (identifier, id)

    train_data = X_train[:, [x_index, y_index]]
    test_data = X_test[:, [x_index, y_index]]

    if x_flip:
        train_data[:, 0] = 1-train_data[:, 0]
        test_data[:, 0] = 1-test_data[:, 0]

    if y_flip:
        train_data[:, 1] = 1-train_data[:, 1]
        test_data[:, 1] = 1-test_data[:, 1]

    phi = Phi(depth, widths, lc_w_range, shift_w_range)
    net = Copula(phi)
    expt(train_data, test_data, net, optim_name,
         optim_args, identifier_id, num_epochs, batch_size, chkpt_freq)


if __name__ == '__main__':
    print('Sample usage: python -m train_scripts.boston.train -F boston_housing')
