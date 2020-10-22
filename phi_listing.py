'''
Contains the standard copula.
For some copula, we use more efficient methods based on
Chapter 2 of 
Matthias, Scherer, and Mai Jan-frederik. Simulating copulas: stochastic models, sampling algorithms, and applications. Vol. 4. World Scientific, 2012.
'''

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import logging


class PhiListing(nn.Module):
    def __init__(self):
        super(PhiListing, self).__init__()

    def inverse(self, t):
        '''
        Return a tensor of nan's by default if no manual inverse is defined.
        '''
        return torch.zeros_like(t) / torch.zeros_like(t)

    def sample(self, ndims, n):
        shape = (n, ndims)
        ms = self.sample_M(n)[:, None].expand(-1, ndims)
        e = torch.distributions.exponential.Exponential(torch.ones(shape))
        E = e.sample()
        return self.forward(E/ms)


class FrankPhi(nn.Module):
    def __init__(self, theta):
        super(FrankPhi, self).__init__()

        self.theta = nn.Parameter(theta)

    def forward(self, t):
        theta = self.theta
        ret = -1/theta * torch.log(torch.exp(-t)*(torch.exp(-theta)-1)+1)
        return ret

    def pdf(self, X):
        return None

    def cdf(self, X):
        return -1./self.theta * \
            torch.log(
                1 + (torch.exp(-self.theta * X[:, 0]) - 1) * (
                    torch.exp(-self.theta * X[:, 1]) - 1) / (torch.exp(-self.theta) - 1)
            )


class ClaytonPhi(PhiListing):
    def __init__(self, theta):
        super(ClaytonPhi, self).__init__()

        self.theta = nn.Parameter(theta)

    def forward(self, t):
        theta = self.theta
        ret = (1+t)**(-1/theta)
        return ret

    def inverse(self, t):
        ret = torch.zeros_like(t) / torch.zeros_like(t)
        ret[torch.abs(t - 1.) < self.eps_snap_zero] = 1.0

        return ret

    def sample_M(self, n):
        alpha = 1./self.theta

        m = torch.distributions.gamma.Gamma(1./self.theta, 1.0)

        return m.sample((n,))

    def pdf(self, X):
        """
        [From Wolfram]
        d/dx((d(x^(-z) + y^(-z) - 1)^(-1/z))/(dy)) = (-1/z - 1) z (-x^(-z - 1)) y^(-z - 1) (x^(-z) + y^(-z) - 1)^(-1/z - 2)
        """
        assert X.shape[1] == 2

        Z = X[:, 0]**(-self.theta) + X[:, 1]**(-self.theta) - 1.
        ret = torch.zeros_like(Z)
        ret[Z > 0] = (-1/self.theta-1.) * self.theta * -X[Z > 0, 0] ** (-self.theta-1) * X[Z > 0, 1] ** (
            -self.theta-1) * (X[Z > 0, 0] ** (-self.theta) + X[Z > 0, 1] ** (-self.theta) - 1) ** (-1./self.theta-2)

        return ret

    def cdf(self, X):
        assert X.shape[1] == 2

        return (torch.max(X[:, 0]**(-self.theta) + X[:, 1]
                          ** (-self.theta)-1, torch.zeros(X.shape[0])))**(-1./self.theta)


class JoePhi(PhiListing):
    """
    The Joe Generator has a derivative that goes to infinity at t = 0. Hence we need
    to be careful when t is close to 0!
    """

    def __init__(self, theta):
        super(JoePhi, self).__init__()

        self.eps = 0
        self.eps_snap_zero = 1e-15
        self.theta = nn.Parameter(theta)

    def forward(self, t):
        eps = self.eps
        if torch.any(t < eps):
            """
            logging.warning('''some entry in t is too small, < %s. May encounter numerical errors if taking gradients.
                            Smallest t= % s. Will be adding eps= % s to inputs for stability.''' % (eps, torch.min(t), eps))
            """
            t_ = t + eps
        else:
            t_ = t + eps
        theta = self.theta
        ret = 1-(1-torch.exp(-t_))**(1/theta) + 1e-7
        return ret

    def inverse(self, t):
        ret = torch.zeros_like(t) / torch.zeros_like(t)
        ret[torch.abs(t - 1.) < self.eps_snap_zero] = 1.0

        return ret

    def sample_M(self, n):
        alpha = 1./self.theta
        U = torch.rand(n)

        ret = torch.ones_like(U)

        ginv_u = self.Ginv(U)
        cond = self.F(torch.floor(ginv_u))

        cut_indices = U <= alpha
        z = cond < U
        j = cond >= U

        ret[z] = torch.ceil(ginv_u[z])
        ret[j] = torch.floor(ginv_u[j])
        ret[cut_indices] = 1.

        return ret

    def Ginv(self, y):
        alpha = 1/self.theta

        return torch.exp(-self.theta * (torch.log(1.-y) + torch.lgamma(1.-alpha)))

    def gamma(self, x):
        return torch.exp(torch.lgamma(x))

    def lbeta(self, x, y):
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x+y)

    def F(self, n):
        alpha = 1/self.theta
        return 1. - 1. / (n * torch.exp(self.lbeta(n, 1.-alpha)))

    def pdf(self, X):
        assert X.shape[1] == 2

        X_ = -X+1.0
        X_1 = X_[:, 0]
        X_2 = X_[:, 1]

        bleh = -X_1 ** (self.theta-1) * X_2 ** (self.theta-1) * \
            ((X_1**self.theta) - (X_1**self.theta - 1) * X_2**self.theta)**(1./self.theta-2) * \
            ((X_1**self.theta-1) * (X_2**self.theta-1) - self.theta)

        return bleh

    def cdf(self, X):
        assert X.shape[1] == 2

        X_ = -X+1.0
        X_1 = X_[:, 0]
        X_2 = X_[:, 1]

        return 1.0 - (X_1**self.theta + X_2**self.theta - (X_1**self.theta)*(X_2**self.theta))**(1./self.theta)


class GumbelPhi(PhiListing):
    def __init__(self, theta):
        super(GumbelPhi, self).__init__()

        self.theta = nn.Parameter(theta)

    def forward(self, t):
        offsetx = 1e-15
        offsety = 1e-15
        theta = self.theta
        ret = torch.exp(-((t+offsetx) ** (1/theta))) + offsety
        return ret

    def pdf(self, X):
        assert X.shape[1] == 2

        u_ = (-torch.log(X[:, 0]))**(self.theta)
        v_ = (-torch.log(X[:, 1]))**(self.theta)

        return torch.exp(-(u_+v_)) ** (1/self.theta)


class IGPhi(nn.Module):
    def __init__(self, theta):
        super(IGPhi, self).__init__()

        self.theta = nn.Parameter(theta)

    def forward(self, t):
        theta = self.theta
        ret = torch.exp((1-torch.sqrt(1+2*theta**2*t))/theta)
        return ret
