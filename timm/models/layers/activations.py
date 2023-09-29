""" Activations

A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn
from torch.nn import functional as F

class LWTA(nn.Module):
    """
        A simple implementation of the LWTA activation to be used as a standalone approach for various models.
    """

    def __init__(self, inplace = True, deterministic = False, U=16, kl = True, temp = 1.):
        super(LWTA, self).__init__()
        self.inplace = inplace
        self.temp_test = 0.01
        self.temp = temp
        self.kl_flag = kl
        self.deterministic = deterministic
        self.U = U

    def forward(self, input):
        #print(input.shape)
        #print(self.U)
        #sys.exit()
        if self.U == 1:
            out = F.gelu(input)
        else:
            out, kl = lwta_activation(input, U = self.U, training = self.training,
                                      temperature = self.temp,
                                      deterministic=self.deterministic,
                                      temp_test = self.temp_test,
                                      kl_flag = self.kl_flag,
                                      )
            self.kl_ = kl

        return out

    def extra_repr(self) -> str:
        inplace_str = 'Competitors: {}, Temperature: {}, Test Temp: {}, Deterministic: {}, KL: {}'.format(self.U, self.temp,
                                                                                                self.temp_test,
                                                                                                self.deterministic,
                                                                                                self.kl_flag)
        return inplace_str


def lwta_activation(input, U = 2, deterministic = False, training = True,
                    temperature = 0.1, temp_test = 0.01, kl_flag = False):
    """
    The general LWTA activation function.
    Can be either deterministic or stochastic depending on the input.
    """

    # case of a fully connected layer
    if len(input.shape) == 2 or len(input.shape) == 3:
        ax = -1 #len(input.shape) - 1
        logits = torch.reshape(input, [-1, input.size(ax)//U, U])

        if deterministic:
            a = torch.argmax(logits, ax, keepdims = True)
            mask_r = torch.zeros_like(logits).scatter_(-1, a, 1.).reshape(input.shape)
        else:
            #print('aaaaa oxi stochasticcccc')
            #sys.exit()
            mask = concrete_sample(logits, temperature if training else temp_test, axis = ax)#.detach()
            mask_r = mask.reshape(input.shape)
    else:
        # this is for convs
        sys.exit()
        x = torch.reshape(input, [-1, input.size(1)//U, U, input.size(-2), input.size(-1)])

        if deterministic:
            a = torch.argmax(x, 2, keepdims = True)
            mask_r = torch.zeros_like(x).scatter_(2, a , 1.).reshape(input.shape)
        else:
            logits = x
            mask = concrete_sample(logits, temperature if training else temp_test, axis = 2)
            mask_r = mask.reshape(input.shape)

    kl = 0.

    if False:#(training and not deterministic) and kl_flag:
        q = F.softmax(logits, -1 if len(input.shape) in [2,3] else 2)
        log_q = torch.log(q + 1e-8)
        log_p = torch.log(torch.tensor(1.0 / U))

        kl = torch.sum(q * (log_q - log_p), -1 if len(input.shape) in [2,3] else 2)
        kl = torch.mean(kl)

    return input*mask_r, kl

def concrete_sample(a, temperature,  eps=1e-8, axis=-1):
    """
    Sample from the concrete relaxation.
    :param probs: torch tensor: probabilities of the concrete relaxation
    :param temperature: float: the temperature of the relaxation
    :param hard: boolean: flag to draw hard samples from the concrete distribution
    :param eps: float: eps to stabilize the computations
    :param axis: int: axis to perform the softmax of the gumbel-softmax trick
    :return: a sample from the concrete relaxation with given parameters
    """
    if temperature == 0.00:
        #logsumexp = torch.logsumexp(a, axis=axis, keepdim=True)
        #log_a = a - logsumexp
        #mask = torch.exp(log_a)
        #return mask
        #max_inds =  torch.argmax(a, axis = axis, keepdims = True)
        #return torch.zeros_like(a).scatter_(-1, max_inds, 1.)
        max_inds = torch.argmax(a, axis, keepdims=True)
        mask_r = torch.zeros_like(a).scatter_(-1, max_inds, 1.)

        return mask_r
    else:
        def _gen_gumbels():
            U = torch.rand(a.shape, device=a.device)
            gumbels = - torch.log(- torch.log(U + eps) + eps)
            if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
                # to avoid zero in exp output
                gumbels = _gen_gumbels()
            return gumbels

        gumbels = _gen_gumbels()  # ~Gumbel(0,1)
        a = (a  + gumbels)/temperature

        #logits_lse = torch.logsumexp(a/temperature, axis = axis, keepdims=True)
        #log_mask = a - logits_lse
        #y_soft = torch.exp(log_mask)

        #gumbels = (a/scale + gumbels) / temperature  # ~Gumbel(logits,tau)
        y_soft = a.softmax(axis)
        #print(gumbels.min(), gumbels.max())
        #ret = y_soft

        index = y_soft.max(axis, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(axis, index, 1.0)
        ret = (y_hard - y_soft).detach() + y_soft
        #ret = y_soft
        #  with some probability use only softmax for the forward pass
        #selection = torch.rand([], device = a.device)
        #ret = torch.where(selection <  select_threshold, y_soft, ret)

        if torch.isnan(ret).sum():
            #import ipdb
            #ipdb.set_trace()
            raise OverflowError(f'gumbel softmax output: {ret}')

        return ret

def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


def mish(x, inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return x.mul(F.softplus(x).tanh())


class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)


def sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()


# PyTorch has this, but not with a consistent inplace argmument interface
class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()


def tanh(x, inplace: bool = False):
    return x.tanh_() if inplace else x.tanh()


# PyTorch has this, but not with a consistent inplace argmument interface
class Tanh(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()


def hard_swish(x, inplace: bool = False):
    inner = F.relu6(x + 3.).div_(6.)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, self.inplace)


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, self.inplace)


def hard_mish(x, inplace: bool = False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardMish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_mish(x, self.inplace)


class PReLU(nn.PReLU):
    """Applies PReLU (w/ dummy inplace arg)
    """
    def __init__(self, num_parameters: int = 1, init: float = 0.25, inplace: bool = False) -> None:
        super(PReLU, self).__init__(num_parameters=num_parameters, init=init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.prelu(input, self.weight)


def gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return F.gelu(x)


class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """
    def __init__(self, inplace: bool = False):
        super(GELU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input)
