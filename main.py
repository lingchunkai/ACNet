import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Function


class PhiInv(nn.Module):
    def __init__(self, phi):
        super(PhiInv, self).__init__()
        self.phi = phi

    def forward(self, y, t0=None, max_iter=400, tol=1e-10):
        with torch.no_grad():
            """
            # We will only run newton's method on entries which do not have
            # a manual inverse defined (via .inverse)
            inverse = self.phi.inverse(y)
            assert inverse.shape == y.shape
            no_inverse_indices = torch.isnan(inverse)
            # print(no_inverse_indices)
            # print(y[no_inverse_indices].shape)
            t_ = newton_root(
                self.phi, y[no_inverse_indices], max_iter=max_iter, tol=tol,
                t0=torch.ones_like(y[no_inverse_indices])*1e-10)

            inverse[no_inverse_indices] = t_
            t = inverse
            """
            t = newton_root(self.phi, y, max_iter=max_iter, tol=tol)

        topt = t.clone().detach().requires_grad_(True)
        f_topt = self.phi(topt)
        return self.FastInverse.apply(y, topt, f_topt, self.phi)

    class FastInverse(Function):
        '''
        Fast inverse function. To avoid running the optimization
        procedure (e.g., Newton's) repeatedly, we pass in the value
        of the inverse (obtained from the forward pass) manually.

        In the backward pass, we provide gradients w.r.t (i) `y`, and
        (ii) `w`, which are any parameters contained in PhiInv.phi. The
        latter is implicitly given by furnishing derivatives w.r.t. f_topt,
        i.e., the function evaluated (with the current `w`) on topt. Note
        that this should contain *values* approximately equal to y, but
        will have the necessary computational graph built up, but detached
        from y.
        '''
        @staticmethod
        def forward(ctx, y, topt, f_topt, phi):
            ctx.save_for_backward(y, topt, f_topt)
            ctx.phi = phi
            return topt

        @staticmethod
        def backward(ctx, grad):
            y, topt, f_topt = ctx.saved_tensors
            phi = ctx.phi

            with torch.enable_grad():
                # Call FakeInverse once again, in order to allow for higher
                # order derivatives to be taken.
                z = PhiInv.FastInverse.apply(y, topt, f_topt, phi)

                # Find phi'(z), i.e., take derivatives of phi(z) w.r.t z.
                f = phi(z)
                dev_z = torch.autograd.grad(f.sum(), z, create_graph=True)[0]

                # To understand why this works, refer to the derivations for
                # inverses. Note that when taking derivatives w.r.t. `w`, we
                # make use of autodiffs automatic application of the chain rule.
                # This automatically finds the derivative d/dw[phi(z)] which
                # when multiplied by the 3rd returned value gives the derivative
                # w.r.t. `w` contained by phi.
                return grad/dev_z, None, -grad/dev_z, None


def newton_root(phi, y, t0=None, max_iter=200, tol=1e-10, guarded=False):
    '''
    Solve
        f(t) = y
    using the Newton's root finding method.

    Parameters
    ----------
    f: Function which takes in a Tensor t of shape `s` and outputs
    the pointwise evaluation f(t).
    y: Tensor of shape `s`.
    t0: Tensor of shape `s` indicating the initial guess for the root.
    max_iter: Positive integer containing the max. number of iterations.
    tol: Termination criterion for the absolute difference |f(t) - y|.
        By default, this is set to 1e-14,
        beyond which instability could occur when using pytorch `DoubleTensor`.
    guarded: Whether we use guarded Newton's root finding method. 
        By default False: too slow and is not necessary most of the time.

    Returns:
        Tensor `t*` of size `s` such that f(t*) ~= y
    '''
    if t0 is None:
        t = torch.zeros_like(y)
    else:
        t = t0.clone().detach()

    s = y.size()
    for it in range(max_iter):

        with torch.enable_grad():
            f_t = phi(t.requires_grad_(True))
            fp_t = torch.autograd.grad(f_t.sum(), t)[0]
            assert not torch.any(torch.isnan(fp_t))

        assert f_t.size() == s
        assert fp_t.size() == s

        g_t = f_t - y

        # Terminate algorithm when all errors are sufficiently small.
        if (torch.abs(g_t) < tol).all():
            break

        if not guarded:
            t = t - g_t / fp_t
        else:
            step_size = torch.ones_like(t)
            for num_guarded_steps in range(1000):
                t_candidate = t - step_size * g_t / fp_t
                f_t_candidate = phi(t_candidate.requires_grad_(True))
                g_candidate = f_t_candidate - y
                overstepped_indices = torch.abs(g_candidate) > torch.abs(g_t)
                if not overstepped_indices.any():
                    t = t_candidate
                    print(num_guarded_steps)
                    break
                else:
                    step_size[overstepped_indices] /= 2.

    assert torch.abs(g_t).max(
    ) < tol, "t=%s, f(t)-y=%s, y=%s, iter=%s, max dev:%s" % (t, g_t, y, it, g_t.max())
    assert t.size() == s
    return t


def bisection_root(phi, y, lb=None, ub=None, increasing=True, max_iter=100, tol=1e-10):
    '''
    Solve
        f(t) = y
    using the bisection method.

    Parameters
    ----------
    f: Function which takes in a Tensor t of shape `s` and outputs
    the pointwise evaluation f(t).
    y: Tensor of shape `s`.
    lb, ub: lower and upper bounds for t.
    increasing: True if f is increasing, False if decreasing.
    max_iter: Positive integer containing the max. number of iterations.
    tol: Termination criterion for the difference in upper and lower bounds.
        By default, this is set to 1e-10,
        beyond which instability could occur when using pytorch `DoubleTensor`.

    Returns:
        Tensor `t*` of size `s` such that f(t*) ~= y
    '''
    if lb is None:
        lb = torch.zeros_like(y)
    if ub is None:
        ub = torch.ones_like(y)

    assert lb.size() == y.size()
    assert ub.size() == y.size()
    assert torch.all(lb < ub)

    f_ub = phi(ub)
    f_lb = phi(lb)
    assert torch.all(
        f_ub >= f_lb) or not increasing, 'Need f to be monotonically non-decreasing.'
    assert torch.all(
        f_lb >= f_ub) or increasing, 'Need f to be monotonically non-increasing.'

    assert (torch.all(
        f_ub >= y) and torch.all(f_lb <= y)) or not increasing, 'y must lie within lower and upper bound. max min y=%s, %s. ub, lb=%s %s' % (y.max(), y.min(), ub, lb)
    assert (torch.all(
        f_ub <= y) and torch.all(f_lb >= y)) or increasing, 'y must lie within lower and upper bound. y=%s, %s. ub, lb=%s %s' % (y.max(), y.min(), ub, lb)

    for it in range(max_iter):
        t = (lb + ub)/2
        f_t = phi(t)

        if increasing:
            too_low, too_high = f_t < y, f_t >= y
            lb[too_low] = t[too_low]
            ub[too_high] = t[too_high]
        else:
            too_low, too_high = f_t > y, f_t <= y
            lb[too_low] = t[too_low]
            ub[too_high] = t[too_high]

        assert torch.all(ub - lb > 0.), "lb: %s, ub: %s" % (lb, ub)

    assert torch.all(ub - lb <= tol)
    return t


def bisection_default_increasing(phi, y):
    '''
    Wrapper for performing bisection method when f is increasing.
    '''
    return bisection_root(phi, y, increasing=True)


def bisection_default_decreasing(phi, y):
    '''
    Wrapper for performing bisection method when f is decreasing.
    '''
    return bisection_root(phi, y, increasing=False)


class MixExpPhi(nn.Module):
    '''
    Sample net for phi involving the sum of 2 negative exponentials.
    phi(t) = m1 * exp(-w1 * t) + m2 * exp(-w2 * t)

    Network Parameters
    ==================
    mix: Tensor of size 2 such that such that (m1, m2) = softmax(mix)
    slope: Tensor of size 2 such that exp(m1) = w1, exp(m2) = w2

    Note that this implies
    i) m1, m2 > 0 and m1 + m2 = 1.0
    ii) w1, w2 > 0
    '''

    def __init__(self, init_w=None):
        import numpy as np
        super(MixExpPhi, self).__init__()

        if init_w is None:
            self.mix = nn.Parameter(torch.tensor(
                [np.log(0.2), np.log(0.8)], requires_grad=True))
            self.slope = nn.Parameter(
                torch.log(torch.tensor([1e1, 1e6], requires_grad=True)))
        else:
            assert len(init_w) == 2
            assert init_w[0].numel() == init_w[1].numel()
            self.mix = nn.Parameter(init_w[0])
            self.slope = nn.Parameter(init_w[1])

    def forward(self, t):
        s = t.size()
        t_ = t.flatten()
        nquery, nmix = t.numel(), self.mix.numel()

        mix_ = torch.nn.functional.softmax(self.mix)
        exps = torch.exp(-t_[:, None].expand(nquery, nmix) *
                         torch.exp(self.slope)[None, :].expand(nquery, nmix))

        ret = torch.sum(mix_ * exps, dim=1)
        return ret.reshape(s)


class MixExpPhi2FixedSlope(nn.Module):
    def __init__(self, init_w=None):
        super(MixExpPhi2FixedSlope, self).__init__()

        self.mix = nn.Parameter(torch.tensor(
            [np.log(0.25)], requires_grad=True))
        self.slope = torch.tensor([1e1, 1e6], requires_grad=True)

    def forward(self, t):
        z = 1./(1+torch.exp(-self.mix[0]))
        return z * torch.exp(-t * self.slope[0]) + (1-z) * torch.exp(-t * self.slope[1])


class Copula(nn.Module):
    def __init__(self, phi):
        super(Copula, self).__init__()
        self.phi = phi
        self.phi_inv = PhiInv(phi)

    def forward(self, y, mode='cdf', others=None, tol=1e-10):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_inv(y, tol=tol)
        cdf = self.phi(inverses.sum(dim=1))

        if mode == 'cdf':
            return cdf

        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Only take gradients with respect to one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur
        elif mode == 'cond_cdf':
            target_dims = others['cond_dims']

            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Only take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator/denominator


def sample(net, ndims, N, seed=142857):
    """
    Note: this does *not* use the efficient method described in the paper.
    Instead, we will use the naive method, i.e., conditioning on each 
    variable in turn and then applying the inverse CDF method on the resultant conditional
    CDF. 

    This method will work on all generators (even those defined by ACNet), and is
    the simplest method assuming no knowledge of the mixing variable M is known.
    """
    # Store old seed and set new seed
    old_rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    U = torch.rand(N, ndims)

    for dim in range(1, ndims):
        print('Sampling from dim: %s' % dim)
        y = U[:, dim].detach().clone()

        def cond_cdf_func(u):
            U_ = U.clone().detach()
            U_[:, dim] = u
            cond_cdf = net(U_[:, :(dim+1)], "cond_cdf",
                           others={'cond_dims': list(range(dim))})
            return cond_cdf

        # Call inverse using the conditional cdf `M` as the function.
        # Note that the weight parameter is set to None since `M` is not parameterized,
        # i.e., hardcoded as the conditional cdf itself.
        U[:, dim] = bisection_default_increasing(cond_cdf_func, y).detach()

    # Revert to old random state.
    torch.random.set_rng_state(old_rng_state)

    return U

####################################################################################
# Tests
####################################################################################


def test_grad_of_phi():
    phi_net = MixExpPhi()
    phi_inv = PhiInv(phi_net)
    query = torch.tensor(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [1., 1., 1.]]).requires_grad_(True)

    gradcheck(phi_net, (query), eps=1e-9)
    gradgradcheck(phi_net, (query,), eps=1e-9)


def test_grad_y_of_inverse():
    phi_net = MixExpPhi()
    phi_inv = PhiInv(phi_net)
    query = torch.tensor(
        [[0.1, 0.2], [0.2, 0.3], [0.25, 0.7]]).requires_grad_(True)

    gradcheck(phi_inv, (query, ), eps=1e-10)
    gradgradcheck(phi_inv, (query, ), eps=1e-10)


def test_grad_w_of_inverse():
    phi_net = MixExpPhi2FixedSlope()
    phi_inv = PhiInv(phi_net)

    eps = 1e-8
    new_phi_inv = copy.deepcopy(phi_inv)

    # Jitter weights in new_phi.
    new_phi_inv.phi.mix.data = phi_inv.phi.mix.data + eps

    query = torch.tensor(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.99, 0.99, 0.99]]).requires_grad_(True)
    old_value = phi_inv(query).sum()
    old_value.backward()
    anal_grad = phi_inv.phi.mix.grad
    new_value = new_phi_inv(query).sum()
    num_grad = (new_value-old_value)/eps

    print('gradient of weights (anal)', anal_grad)
    print('gradient of weights (num)', num_grad)


def test_grad_y_of_pdf():
    phi_net = MixExpPhi()
    query = torch.tensor(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.99, 0.99, 0.99]]).requires_grad_(True)
    cop = Copula(phi_net)
    def f(y): return cop(y, mode='pdf')
    gradcheck(f, (query, ), eps=1e-8)
    # This fails sometimes if rtol is too low..?
    gradgradcheck(f, (query, ), eps=1e-8, atol=1e-6, rtol=1e-2)


def plot_pdf_and_cdf_over_grid():
    phi_net = MixExpPhi()
    cop = Copula(phi_net)

    n = 500
    x1 = np.linspace(0.001, 1, n)
    x2 = np.linspace(0.001, 1, n)
    xv1, xv2 = np.meshgrid(x1, x2)
    xv1_tensor = torch.tensor(xv1.flatten())
    xv2_tensor = torch.tensor(xv2.flatten())
    query = torch.stack((xv1_tensor, xv2_tensor)
                        ).double().t().requires_grad_(True)
    cdf = cop(query, mode='cdf')
    pdf = cop(query, mode='pdf')

    assert abs(pdf.mean().detach().numpy().sum() -
               1) < 1e-6, 'Mean of pdf over grid should be 1'
    assert abs(cdf[-1].detach().numpy().sum() -
               1) < 1e-6, 'CDF at (1..1) should be should be 1'


def plot_cond_cdf():
    phi_net = MixExpPhi()
    cop = Copula(phi_net)

    n = 500
    xv2 = np.linspace(0.001, 1, n)
    xv2_tensor = torch.tensor(xv2.flatten())
    xv1_tensor = 0.9 * torch.ones_like(xv2_tensor)
    x = torch.stack([xv1_tensor, xv2_tensor], dim=1).requires_grad_(True)
    cond_cdf = cop(x, mode="cond_cdf", others={'cond_dims': [0]})

    plt.figure()
    plt.plot(cond_cdf.detach().numpy())
    plt.title('Conditional CDF')
    plt.draw()
    plt.pause(0.01)


def plot_samples():
    phi_net = MixExpPhi()
    cop = Copula(phi_net)

    s = sample(cop, 2, 2000, seed=142857)
    s_np = s.detach().numpy()

    plt.figure()
    plt.scatter(s_np[:, 0], s_np[:, 1])
    plt.title('Sampled points from Copula')
    plt.draw()
    plt.pause(0.01)


def plot_loss_surface():
    phi_net = MixExpPhi2FixedSlope()
    cop = Copula(phi_net)

    s = sample(cop, 2, 2000, seed=142857)
    s_np = s.detach().numpy()

    l = []
    x = np.linspace(-1e-2, 1e-2, 1000)
    for SS in x:
        new_cop = copy.deepcopy(cop)
        new_cop.phi.mix.data = cop.phi.mix.data + SS

        loss = -torch.log(new_cop(s, mode='pdf')).sum()
        l.append(loss.detach().numpy().sum())

    plt.figure()
    plt.plot(x, l)
    plt.title('Loss surface')
    plt.draw()
    plt.pause(0.01)


def test_training(test_grad_w=False):
    gen_phi_net = MixExpPhi()
    gen_phi_inv = PhiInv(gen_phi_net)
    gen_cop = Copula(gen_phi_net)

    s = sample(gen_cop, 2, 2000, seed=142857)
    s_np = s.detach().numpy()

    ideal_loss = -torch.log(gen_cop(s, mode='pdf')).sum()

    train_cop = copy.deepcopy(gen_cop)
    train_cop.phi.mix.data *= 1.5
    train_cop.phi.slope.data *= 1.5
    print('Initial loss', ideal_loss)
    optimizer = optim.Adam(train_cop.parameters(), lr=1e-3)

    def numerical_grad(cop):
        # Take gradients w.r.t to the first mixing parameter
        print('Analytic gradients:', cop.phi.mix.grad[0])

        old_cop, new_cop = copy.deepcopy(cop), copy.deepcopy(cop)
        # First order approximation of gradient of weights
        eps = 1e-6
        new_cop.phi.mix.data[0] = cop.phi.mix.data[0] + eps
        x2 = -torch.log(new_cop(s, mode='pdf')).sum()
        x1 = -torch.log(cop(s, mode='pdf')).sum()

        first_order_approximate = (x2-x1)/eps
        print('First order approx.:', first_order_approximate)

    for iter in range(100000):
        optimizer.zero_grad()
        loss = -torch.log(train_cop(s, mode='pdf')).sum()
        loss.backward()
        print('iter', iter, ':', loss, 'ideal loss:', ideal_loss)
        if test_grad_w:
            numerical_grad(train_cop)
        optimizer.step()


if __name__ == '__main__':
    import torch.optim as optim
    from torch.autograd import gradgradcheck, gradcheck
    import numpy as np
    import logging as log
    import matplotlib.pyplot as plt
    import copy

    torch.set_default_tensor_type(torch.DoubleTensor)

    test_grad_of_phi()
    test_grad_y_of_inverse()
    test_grad_w_of_inverse()
    test_grad_y_of_pdf()

    plot_pdf_and_cdf_over_grid()
    plot_cond_cdf()
    plot_samples()
    """ Uncomment for rudimentary training. 
        Note: very slow and unrealistic.
    plot_loss_surface()
    test_training()
    """
