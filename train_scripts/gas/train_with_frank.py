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

month1_path = "data/gas/batch1.dat"
# Used if we want to compare between months.
month2_path = "data/gas/batch2.dat"


# Extract data which compares between features.
def data_test_between_features(feature_ids, sensor_id, month_data):
    # assert len(feature_ids) == 2

    num_features = len(feature_ids)

    features = []
    for feature_id in feature_ids:

        full_feature_id = feature_id + sensor_id * 8
        if feature_id in [5, 6, 7]:
            features.append(-month_data[:, full_feature_id])
        else:
            features.append(month_data[:, full_feature_id])

    return features


# Extract data which compares between sensor ids.
def data_test_between_sensors(feature_id, sensor_ids, month_data):
    assert len(sensor_ids) == 2

    id1 = feature_id[0] + sensor_ids[0] * 8
    id2 = feature_id[1] + sensor_ids[1] * 8
    return month_data[:, id1], month_data[:, id2]


# Extract data which compares between different months.
def data_test_between_months(feature_id, sensor_id, months_data):
    assert len(months_data) == 2

    id = feature_id + sensor_id * 8
    return months_data[0][:, id], months_data[1][:, id]


def read_batch(filepath):
    def format_feature(x):
        return float(x.decode('UTF-8').split(':')[1])

    d = [(i, format_feature) for i in range(1, 129)]
    z = np.genfromtxt(filepath,
                      delimiter=" ",
                      usecols=list(range(1, 129)),
                      converters=dict(d),
                      )
    return z


month1 = read_batch(month1_path)
month2 = read_batch(month2_path)

identifier = 'gas_2012_frank'
ex = Experiment('gas_2012_frank')

torch.set_default_tensor_type(torch.DoubleTensor)

data = data_test_between_features((0, 4, 7), 0, month1)
data = data_test_between_features((0, 4, 7), 2, month1)
d1 = data[0]
d2 = data[1]
d3 = data[2]

X = np.concatenate([d1[:, None], d2[:, None], d3[:, None]], axis=1)

# plt.scatter(X[:, 0], X[:, 1])
# plt.show()


def add_train_random_noise(data, num_adds):
    new_data = np.random.rand(num_adds, data.shape[1])
    print(data.shape)
    print(new_data.shape)
    return np.concatenate((data, new_data), axis=0)


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


# plt.scatter(X_train[:, 0], X_train[:, 1])
# plt.show()


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
    batch_size = 200
    chkpt_freq = 500

    from phi_listing import FrankPhi
    Phi = FrankPhi
    phi_name = 'FrankPhi'

    # Initial parameters.
    initial_theta = 5.

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

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_data, batch_size=1000000, shuffle=True)

    optimizer = get_optim(optim_name, net, optim_args)

    train_loss_per_epoch = []

    for epoch in range(num_epochs):
        print(net.phi.theta)
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
def run(Phi,
        initial_theta,
        optim_name, optim_args,
        num_epochs, batch_size, chkpt_freq,
        frac_rand):
    id = get_info()
    identifier_id = '%s%s' % (identifier, id)

    train_data = X_train
    train_data = add_train_random_noise(train_data, int(X_train.shape[0]*frac_rand))
    test_data = X_test

    phi = Phi(torch.tensor(initial_theta))
    net = Copula(phi)
    expt(train_data, test_data, net, optim_name,
         optim_args, identifier_id, num_epochs, batch_size, chkpt_freq)


if __name__ == '__main__':
    print('Sample usage: python -m train_scripts.gas.train_with_frank -F learn_gas_with_frank')
