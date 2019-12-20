# pylint: disable=missing-docstring, no-member, invalid-name, arguments-differ
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def swish_jit_fwd(x):
    return x * torch.sigmoid(x) * 1.6768


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid))) * 1.6768


class SwishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return swish_jit_bwd(x, grad_output)


def swish(x):
    return SwishJitAutoFn.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


def orthogonal_(tensor, gain=1):
    # proper orthogonal init, see https://github.com/pytorch/pytorch/pull/10672
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new_empty(rows, cols).normal_(0, 1)

    for i in range(0, rows, cols):
        # Compute the qr factorization
        q, r = torch.qr(flattened[i:i + cols].t())
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        q *= torch.diag(r, 0).sign()
        q.t_()

        with torch.no_grad():
            tensor[i:i + cols].view_as(q).copy_(q)

    with torch.no_grad():
        tensor.mul_(gain * cols ** 0.5)
    return tensor


class Conv(nn.Module):
    def __init__(self, n_in, n_out, beta, k=3, padding=None, groups=1, **kargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in // groups, k, k))
        self.omega = (n_in // groups * k ** 2) ** -0.5
        orthogonal_(self.weight)

        self.bias = nn.Parameter(torch.zeros(n_out))
        self.beta = beta

        if padding is None:
            padding = k // 2
        kargs['padding'] = padding
        kargs['groups'] = groups
        self.kargs = kargs

    def forward(self, x):
        return F.conv2d(x, self.omega * self.weight, self.beta * self.bias, **self.kargs)


class Linear(nn.Module):
    def __init__(self, n_in, n_out, beta):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in))
        self.omega = n_in ** -0.5
        orthogonal_(self.weight)

        self.bias = nn.Parameter(torch.zeros(n_out))
        self.beta = beta

    def forward(self, x):
        return F.linear(x, self.omega * self.weight, self.beta * self.bias)


class Memory:
    def __init__(self, x=None):
        self.x = [] if x is None else [x]
    def __call__(self, x=None):
        if x is not None:
            self.x.insert(0, x)
        return self.x[0]
    def __getitem__(self, i):
        return self.x[i]


class MLP(nn.Sequential):
    def __init__(self, L, n, beta):
        super().__init__()

        m = Memory(3 * 32 * 32)

        for i in range(L):
            self.add_module('linear%d' % (i + 1), Linear(m(), m(n), beta))
            self.add_module('act%d' % (i + 1), Swish())

        self.add_module('classifier', Linear(m(), m(10), beta))

    def forward(self, x):
        return super().forward(x.flatten(1))


def _mlp(arch, L, n, beta, pretrained, progress, **kwargs):
    model = MLP(L, n, beta, **kwargs)
    if pretrained:
        url = "https://github.com/mariogeiger/PyTorch-CIFAR10/releases/download/1.3/{}.pt".format(arch)
        from torch.hub import load_state_dict_from_url
        state_dict = load_state_dict_from_url(url, map_location='cpu', progress=progress)
        model.load_state_dict(state_dict)
    return model


def mlp5(pretrained=False, progress=True, **kwargs):
    r"""MultiLayerPerceptron

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _mlp('mlp5', 5, 512, 0.1, pretrained, progress, **kwargs)


class ConvNet(nn.Module):
    def __init__(self, n, L1, L2, L3, beta):
        super().__init__()

        C = partial(Conv, beta=beta)
        m = Memory(3)

        # (L1 + L2 + L3) * 2 + 7

        seq = [
            nn.Sequential(
                C(m(), m(n), k=1),
                Swish(),
                C(m(), m(), k=3, groups=m()),
                Swish(),
            )
            for _ in range(L1)
        ]

        seq += [
            nn.Sequential(
                C(m(), m(), k=1),
                Swish(),
                C(m(), m(), k=3, groups=m(), stride=2),  # 32 -> 16
                Swish(),
            ),
        ]

        seq += [
            nn.Sequential(
                C(m(), m(2*n), k=1),
                Swish(),
                C(m(), m(), k=3, groups=m()),
                Swish(),
            )
            for _ in range(L2)
        ]

        seq += [
            nn.Sequential(
                C(m(), m(), k=1),
                Swish(),
                C(m(), m(), k=3, groups=m(), stride=2), # 32 -> 8
                Swish(),
            ),
        ]

        seq += [
            nn.Sequential(
                C(m(), m(4*n), k=1),
                Swish(),
                C(m(), m(), k=3, groups=m()),
                Swish(),
            )
            for _ in range(L3)
        ]

        seq += [
            nn.Sequential(
                C(m(), m(), k=1),
                Swish(),
                C(m(), m(), k=3, groups=m(), stride=2), # 8 -> 4
                Swish(),
            ),
            nn.Sequential(
                C(m(), m(8*n), k=1),
                Swish(),
            ),

            nn.AdaptiveAvgPool2d(1)
        ]

        self.seq = nn.Sequential(*seq)

        self.classifier = Linear(m(), m(10), beta)

    def forward(self, x):
        x = self.seq(x).flatten(1)
        return self.classifier(x)


def _convnet(arch, n, L1, L2, L3, beta, pretrained, progress):
    model = ConvNet(n, L1, L2, L3, beta)
    if pretrained:
        url = "https://github.com/mariogeiger/PyTorch-CIFAR10/releases/download/1.3/{}.pt".format(arch)
        from torch.hub import load_state_dict_from_url
        state_dict = load_state_dict_from_url(url, map_location='cpu', progress=progress)
        model.load_state_dict(state_dict)
    return model


def convnet23(pretrained=False, progress=True):
    r"""ConvNet

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _convnet('convnet23', 32, 2, 5, 1, 0.1, pretrained, progress)


def convnet31(pretrained=False, progress=True):
    r"""ConvNet

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _convnet('convnet31', 48, 3, 7, 2, 0.1, pretrained, progress)


def convnet43(pretrained=False, progress=True):
    r"""ConvNet

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _convnet('convnet43', 64, 5, 10, 3, 0.1, pretrained, progress)
