import imp
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import MultiStepLR
from accuracy import Accuracy


class CIFAR10Module(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        image, label = batch
        predictions = self.model(image)

        loss = self.criterion(predictions, label)

        accuracy = self.train_acc(predictions.max(1)[1], label)
        self.log("loss_train", loss)
        self.log("accuracy/train", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        predictions = self.model(image)
        accuracy = self.val_acc(predictions.max(1)[1], label)
        self.log("accuracy/val", accuracy)

    def test_step(self, batch, batch_idx):
        image, label = batch
        predictions = self.model(image)
        accuracy = self.val_acc(predictions.max(1)[1], label)
        self.log("accuracy/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        milestones = [
            # int(0.25 * self.hparams.max_epochs),
            int(0.50 * self.hparams.max_epochs),
            int(0.75 * self.hparams.max_epochs),
        ]
        scheduler = {
            "scheduler": MultiStepLR(optimizer, milestones=milestones, gamma=0.1),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]
