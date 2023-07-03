import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from schduler import WarmupCosineLR

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl

# from vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
}

class CIFAR10Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        self.save_hyperparameters(hparams)

        if hparams.classifier == "vgg11_bn":
            self.model = vgg11_bn(pretrained=False, progress=True, device=self.device)
        elif hparams.classifier == "vgg13_bn":
            self.model = vgg13_bn(pretrained=False, progress=True, device=self.device)
        elif hparams.classifier == "vgg16_bn":
            self.model = vgg16_bn(pretrained=False, progress=True, device=self.device)
        elif hparams.classifier == "vgg19_bn":
            self.model = vgg19_bn(pretrained=False, progress=True, device=self.device)
        else:
            raise ValueError("Invalid classifier name.")

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = StepLR(
            optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log("loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        acc = correct / len(x)
        self.log("loss/val", loss, prog_bar=True)
        self.log("acc/val", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        acc = correct / len(x)
        self.log("loss/test", loss, prog_bar=True)
        self.log("acc/test", acc, prog_bar=True)
