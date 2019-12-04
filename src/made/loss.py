"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Training 2 signals together
"""

import os
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    # Pytorch ordering
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss_1d(target, params, emd=False):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
        target = [N, D]
        params = [N, D, 3K]
    """
    gap = 1. / 255. # any gap is ok, as we optimize the scalar at denominator doesnt matter
    x = target
    l = params
    # Pytorch ordering
    # x = x.permute(0, 2, 3, 1)
    # l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    D = xs[-1] # num of points
    # K
    nr_mix = int(ls[-1] / 3)
    # [N, D, K ]
    logit_probs = l[:, :, :nr_mix]
    # [N, D, 2K]
    l = l[:, :, nr_mix:].contiguous() # 2 for mean, scale
    # [N, D, K]
    means = l[:, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    # [N, D, K]
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)
    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + gap)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - gap)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    # log(sig(x)) = -sp(-x) = x - sp(x)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    # log(1 - sg(x)) = log(sg(-x)) = -sp(x)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    # [N, D, K]
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs        = log_probs + log_prob_from_logits(logit_probs)

    kl = -torch.sum(log_sum_exp(log_probs))

    if emd:
        # weighted mean => form a pc
        emd_loss = torch.norm(centered_x, dim=1, p=2)
        emd_loss = emd_loss.mean()
    else:
        emd_loss = 0

    return kl, emd_loss


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda : one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)


def sample_from_discretized_mix_logistic_1d(params, nr_mix):
    # Pytorch ordering
    # l = l.permute(0, 2, 3, 1)
    l = params #[N, D, K]
    ls = [int(y) for y in l.size()]
    # xs = [N, D, 1]
    xs = ls[:-1] + [1] #[3]

    # unpack parameters
    logit_probs = l[:, :, :nr_mix]
    l = l[:, :, nr_mix:].contiguous() # for mean, scale

    # sample mixture indicator from softmax
    # [N, D, K]
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda : temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    # [N, D, 1]
    _, argmax = temp.max(dim=2)

    # [N, D, K]
    one_hot = to_one_hot(argmax, nr_mix)
    # [N, D, 1, K]
    # sel = one_hot.view(xs[:-1] + [1, nr_mix])
    sel = one_hot.view(xs[:-1] + [nr_mix])

    # select logistic parameters
    means = torch.sum(l[:, :, :nr_mix] * sel, dim=2)
    log_scales = torch.clamp(torch.sum(
        l[:, :, nr_mix:2 * nr_mix] * sel, dim=2), min=-7.)

    u = torch.FloatTensor(means.size())
    if l.is_cuda : u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)

    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x, min=-1.), max=1.)
    #
    # out = x0.unsqueeze(1)
    out = x0

    return out


''' utilities for shifting the image around, efficient alternative to masking convolutions '''
def down_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when downshifting, the last row is removed
    x = x[:, :, :xs[2] - 1, :]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


def right_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when righshifting, the last column is removed
    x = x[:, :, :, :xs[3] - 1]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)


def load_part_of_model(model, path):
    params = torch.load(path)
    added = 0
    for name, param in params.items():
        if name in model.state_dict().keys():
            try :
                model.state_dict()[name].copy_(param)
                added += 1
            except Exception as e:
                print(e)
                pass
    print('added %s of params:' % (added / float(len(model.state_dict().keys()))))
