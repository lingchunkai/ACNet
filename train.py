'''
Helper files for training.
'''

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


def load_data(path, num_train, num_test):
    '''
    Loads dataset from `path` split into Pytorch train and test of 
    given sizes. Train set is taken from the front while
    test set is taken from behind.

    :param path: path to .p file containing data.
    '''
    f = open(path, 'rb')
    all_data = pickle.load(f)['samples']

    ndata_all = all_data.size()[0]
    assert num_train+num_test <= ndata_all

    train_data = all_data[:num_train]
    test_data = all_data[(ndata_all-num_test):]

    return train_data, test_data


def load_log_ll(path, num_train, num_test):
    f = open(path, 'rb')
    all_log_ll = pickle.load(f)['log_ll']

    ndata_all = all_log_ll.numel()
    assert num_train+num_test <= ndata_all

    train_log_ll = all_log_ll[:num_train]
    test_log_ll = all_log_ll[(ndata_all-num_test):]

    return train_log_ll, test_log_ll


def get_optim(name, net, args):
    if name == 'SGD':
        optimizer = optim.SGD(net.parameters(), args['lr'], args['momentum'])
    elif name == 'Adam':
        # TODO: add in more. Note: we do not use this in the paper.
        optimizer = optim.Adam(net.parameters(), args['lr'])
    elif name == 'RMSprop':
        # TODO: add in more. Note: we do not use this in the paper.
        optimizer = optim.RMSprop(net.parameters(), args['lr'])

    return optimizer


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

    # IMPORTANT: for this experiment, we did *not* perform hyperparameter tuning.
    # Hence, the `validation loss' here is essentially `test` loss.
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

            """
            if args.dims == 2:
                print('Scatter sampling')
                samples = sample(net, 2, 1000)
                plt.scatter(samples[:, 0], samples[:, 1])
                plt.savefig('./sample_figs/%s/epoch%s.png' %
                            (identifier, epoch))
                plt.clf()
            else:
                print('Not doign scatter plot, dims > 2')
            """

            print('Evaluating validation/test loss.')
            for j, val_data in enumerate(val_loader, 0):
                net.zero_grad()
                val_p = net(val_data, mode='pdf')
                val_loss = -torch.mean(torch.log(val_p))
            print('Average validation/test loss %s' % val_loss)


def make_ranged_data(data, width, seed):
    '''
    Transforms each coordinate (x, y) to 
    the range ([x-e1, x+e2], [y-e3, y+e4])
    where e1, e2, e3, e4 are drawn uniformly from [0, width].
    The final ranges are snapped to [0, 1].
    '''

    assert data.shape[1] == 2

    np.random.seed(seed)

    epsilon_lower = np.random.random_sample((data.shape)) * width
    epsilon_upper = np.random.random_sample((data.shape)) * width

    epsilon_lower = torch.from_numpy(epsilon_lower)
    epsilon_upper = torch.from_numpy(epsilon_upper)

    bounds_lower = torch.max(torch.zeros_like(data), data - epsilon_lower)
    bounds_upper = torch.min(torch.ones_like(data), data + epsilon_upper)

    return bounds_lower, bounds_upper


def expt_cdf_noisy(train_data, val_data,
                   net,
                   optim_name,
                   optim_args,
                   identifier,
                   width,
                   seed,
                   num_epochs=1000,
                   batch_size=100,
                   chkpt_freq=50,
                   ):
    '''
    Add in uncertainty in all points
    '''

    os.mkdir('./checkpoints/%s' % identifier)
    os.mkdir('./sample_figs/%s' % identifier)

    train_bounds_lower, train_bounds_upper = make_ranged_data(
        train_data, width, seed)
    val_bounds_lower, val_bounds_upper = make_ranged_data(
        val_data, width, seed)

    train_bounds = torch.cat(
        [train_bounds_lower, train_bounds_upper], dim=1)
    val_bounds = torch.cat(
        [val_bounds_lower, val_bounds_upper], dim=1)

    train_loader = DataLoader(
        train_bounds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_data, batch_size=1000000, shuffle=True)

    optimizer = get_optim(optim_name, net, optim_args)

    train_loss_per_epoch = []

    for epoch in range(num_epochs):
        loss_per_minibatch = []
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()

            d = torch.tensor(data, requires_grad=True)
            dsize = d.shape[0]

            big = data[:, 2:]
            small = data[:, 0:2]
            cross1 = torch.cat(
                [data[:, 0:1], data[:, 3:4]], dim=1)
            cross2 = torch.cat(
                [data[:, 2:3], data[:, 1:2]], dim=1)

            joint = torch.cat([big, small, cross1, cross2], dim=0)
            P_raw = net(torch.tensor(joint, requires_grad=True), mode='cdf')
            P_big = P_raw[:dsize]
            P_small = P_raw[dsize:(2*dsize)]
            P_cross1 = P_raw[(2*dsize):(3*dsize)]
            P_cross2 = P_raw[(3*dsize):(4*dsize)]
            P = P_big + P_small - P_cross1 - P_cross2

            logloss = -torch.sum(torch.log(P))
            reg_loss = logloss
            reg_loss.backward()
            scalar_loss = (reg_loss/P.numel()).detach().numpy().item()

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

            """
            if args.dims == 2:
                print('Scatter sampling')
                samples = sample(net, 2, 1000)
                plt.scatter(samples[:, 0], samples[:, 1])
                plt.savefig('./sample_figs/%s/epoch%s.png' %
                            (identifier, epoch))
                plt.clf()
            else:
                print('Not doign scatter plot, dims > 2')
            """

            print('Evaluating validation loss')
            for j, val_data in enumerate(val_loader, 0):
                net.zero_grad()
                val_p = net(val_data, mode='pdf')
                val_loss = -torch.mean(torch.log(val_p))
            print('Average validation loss %s' % val_loss)
