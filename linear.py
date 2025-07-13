import math

import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F


class BayesLinear(Module):
    r"""
    Applies Bayesian Linear

    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.

    """
    __constants__ = ['prior_mu', 'prior_sigma', 'bias', 'in_features', 'out_features']

    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias=False):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))

        # self.weight = self.weight_mu     # 为了使代码正常运行
        self.register_buffer('weight_eps', None)

        if bias is None or bias is False:
            self.bias = False
        else:
            self.bias = True


        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = Parameter(torch.Tensor(out_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)

        self.reset_parameters()

    def reset_parameters(self):
        # # Initialization method of Adv-BNN
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        # # self.weight_mu.data.uniform_(-stdv, stdv)
        # init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # # self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        # self.weight_log_sigma.data.normal_(mean=-5, std=0.1)
        n = max(self.weight_mu.size(1),self.weight_mu.size(0))  # 假设 weight_mu 的形状是 [out_features, in_features]
        nn.init.uniform_(self.weight_mu,
                         -math.sqrt(6 / n), math.sqrt(6 / n))
        ε = 0.05
        G = torch.empty(self.weight_log_sigma.size()).uniform_(ε / math.sqrt(2), ε)
        self.weight_log_sigma.data.copy_(torch.log(G ** 2))

        if self.bias:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

        # Initialization method of the original torch nn.linear.

    #         init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
    #         self.weight_log_sigma.data.fill_(self.prior_log_sigma)

    #         if self.bias :
    #             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
    #             bound = 1 / math.sqrt(fan_in)
    #             init.uniform_(self.bias_mu, -bound, bound)

    #             self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def freeze(self):
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        if self.bias:
            self.bias_eps = torch.randn_like(self.bias_log_sigma)

    def unfreeze(self):
        self.weight_eps = None
        if self.bias:
            self.bias_eps = None

    def forward(self, input):
        r"""
        Overriden.
        """

        if self.weight_eps is None:
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        else:
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps

        if self.bias:
            if self.bias_eps is None:
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else:
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps
        else:
            bias = None
        weight = weight.to(input.device)
        if bias is not None:
            bias = bias.to(input.device)
        return F.linear(input, weight, bias)

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu,
                                                                                              self.prior_sigma,
                                                                                              self.in_features,
                                                                                              self.out_features,
                                                                                              self.bias is not None)

