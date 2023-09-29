""" LWTA Activations"""

import torch
from torch import nn as nn
from torch.nn import functional as F

class LWTA(nn.Module):
    """
        A simple implementation of the LWTA activation to be used as a standalone approach for various models.
    """

    def __init__(self, inplace = True, deterministic = False, U=16, kl = True, temp = 1.67):
        super(LWTA, self).__init__()
        self.inplace = inplace
        self.temp_test = 1.67
        self.temp = temp
        self.kl_flag = kl
        self.deterministic = deterministic
        self.U = U

    def forward(self, input):

        if self.U == 1:
            return nn.functional.relu(input)
        else:
            out, kl = lwta_activation(input, U=self.U, training=self.training,
                                      temperature=self.temp,
                                      deterministic=self.deterministic,
                                      temp_test=self.temp_test,
                                      kl_flag=self.kl_flag,
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
        #sys.exit()
        x = torch.reshape(input, [-1, input.size(1)//U, U, input.size(-2), input.size(-1)])

        if deterministic:
            a = torch.argmax(x, 2, keepdims = True)
            mask_r = torch.zeros_like(x).scatter_(2, a , 1.).reshape(input.shape)
        else:
            logits = x
            mask = concrete_sample(logits, temperature if training else temp_test, axis = 2)
            mask_r = mask.reshape(input.shape)
            #mask = concrete_sample(logits.mean([-2,-1], keepdim = True), temperature if training else temp_test, axis = 2)
            #mask_r = mask.reshape([-1, input.size(1)//U*U, 1, 1])

    kl = 0.

    if False:#(training and not deterministic) and kl_flag:
        q = F.softmax(logits, -1 if len(input.shape) in [2,3] else 2)
        log_q = torch.log(q + 1e-8)
        log_p = torch.log(torch.tensor(1.0 / U))

        kl = torch.sum(q * (log_q - log_p), -1 if len(input.shape) in [2,3] else 2)
        kl = torch.mean(kl)

    return input*mask_r, kl

def concrete_sample(a, temperature,  eps=1e-5, axis=-1):
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

        y_soft = a.softmax(axis)

        index = y_soft.max(axis, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(axis, index, 1.0)
        ret = (y_hard - y_soft).detach() + y_soft

        if torch.isnan(ret).sum():
            raise OverflowError(f'gumbel softmax output: {ret.max(), ret.min()}')

        return ret

