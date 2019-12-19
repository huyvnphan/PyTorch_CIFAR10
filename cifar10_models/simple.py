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


class Conv(nn.Module):
    def __init__(self, n_in, n_out, beta, k=3, padding=None, groups=1, **kargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in // groups, k, k))
        self.omega = (n_in // groups * k ** 2) ** -0.5

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

        self.bias = nn.Parameter(torch.zeros(n_out))
        self.beta = beta

    def forward(self, x):
        return F.linear(x, self.omega * self.weight, self.beta * self.bias)


class Memory:
    def __init__(self, x=None):
        self.x = x
    def __call__(self, x=None):
        if x is not None:
            self.x = x
        return self.x


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
    def __init__(self, beta):
        super().__init__()

        C = partial(Conv, beta=beta)
        m = Memory(3)

        self.seq = nn.Sequential(
            C(m(), m(16), k=1),
            Swish(),
            C(m(), m(), k=3, groups=m(), stride=2),  # 32 -> 16
            Swish(),

            C(m(), m(32), k=1),
            Swish(),
            C(m(), m(), k=3, groups=m()),
            Swish(),

            C(m(), m(), k=1),
            Swish(),
            C(m(), m(), k=3, groups=m()),
            Swish(),

            C(m(), m(), k=1),
            Swish(),
            C(m(), m(), k=3, groups=m(), stride=2), # 32 -> 8
            Swish(),

            C(m(), m(64), k=1),
            Swish(),
            C(m(), m(), k=3, groups=m(), stride=2), # 8 -> 4
            Swish(),

            C(m(), m(128), k=1),
            Swish(),
            C(m(), m(), k=3, groups=m()),
            Swish(),

            C(m(), m(512), k=1),
            Swish(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = Linear(m(), m(10), beta)

    def forward(self, x):
        x = self.seq(x).flatten(1)
        return self.classifier(x)


def _convnet(arch, beta, pretrained, progress):
    model = ConvNet(beta)
    if pretrained:
        url = "https://github.com/mariogeiger/PyTorch-CIFAR10/releases/download/1.3/{}.pt".format(arch)
        from torch.hub import load_state_dict_from_url
        state_dict = load_state_dict_from_url(url, map_location='cpu', progress=progress)
        model.load_state_dict(state_dict)
    return model


def convnet(pretrained=False, progress=True):
    r"""ConvNet

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _convnet('convnet', 0.1, pretrained, progress)
