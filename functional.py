import math
import torch

from loss import *
from linear import *
def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1) :
    """
    An method for calculating KL divergence between two Normal distribtuion.

    Arguments:
        mu_0 (Float) : mean of normal distribution.
        log_sigma_0 (Float): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).

    """
    if isinstance(log_sigma_1, torch.Tensor):
        kl = log_sigma_1 - log_sigma_0 + \
        (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*torch.exp(log_sigma_1)**2) - 0.5
    else:
        kl = log_sigma_1 - log_sigma_0 + \
        (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*math.exp(log_sigma_1)**2) - 0.5

    return kl.sum()

def bayesian_kl_loss(model, reduction='mean', last_layer_only=False,prior_model=None) :
    """
    An method for calculating KL divergence of whole layers in the model.


    Arguments:
        model (nn.Module): a model to be calculated for KL-divergence.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.

    """
    # device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    if prior_model is None:
        device = next(model.parameters()).device
        kl = torch.Tensor([0]).to(device)
        kl_sum = torch.Tensor([0]).to(device)
        n = torch.Tensor([0]).to(device)

        for m in model.modules() :
            if isinstance(m, BayesLinear):
                kl = _kl_loss(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma).to(device)
                kl_sum += kl
                n += len(m.weight_mu.view(-1))

                if m.bias :
                    kl = _kl_loss(m.bias_mu, m.bias_log_sigma, m.prior_mu, m.prior_log_sigma)
                    kl_sum += kl
                    n += len(m.bias_mu.view(-1))



        if last_layer_only or n == 0 :
            return kl

        if reduction == 'mean' :
            return kl_sum/n
        elif reduction == 'sum' :
            return kl_sum
        else :
            raise ValueError(reduction + " is not valid")

    else:
        device = next(model.parameters()).device
        kl = torch.Tensor([0]).to(device)
        kl_sum = torch.Tensor([0]).to(device)
        n = torch.Tensor([0]).to(device)
        # current_model=model.module
        current_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        prior_model = prior_model.to(device)
        for (m, prior_m) in zip(current_model.modules(), prior_model.modules()):
            # 检查是否为贝叶斯层
            if isinstance(m, BayesLinear):
                # 计算权重的KL散度
                kl = _kl_loss(
                    m.weight_mu, m.weight_log_sigma,
                    prior_m.weight_mu, prior_m.weight_log_sigma
                ).to(device)
                kl_sum += kl
                n += len(m.weight_mu.view(-1))

                # 如果有偏置，计算偏置的KL散度
                if hasattr(m, 'bias_mu') and m.bias_mu is not None:
                    kl = _kl_loss(
                        m.bias_mu, m.bias_log_sigma,
                        prior_m.bias_mu, prior_m.bias_log_sigma
                    ).to(device)
                    kl_sum += kl
                    n += len(m.bias_mu.view(-1))

        if last_layer_only or n == 0:
            return kl

        if reduction == 'mean':
            return kl_sum / n
        elif reduction == 'sum':
            return kl_sum
        else:
            raise ValueError(reduction + " is not valid")


