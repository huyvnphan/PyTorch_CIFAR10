import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from cifar10_models import *

def get_classifier(classifier):
    if classifier == 'vgg11_bn':
        return vgg11_bn()
    elif classifier == 'vgg13_bn':
        return vgg13_bn()
    elif classifier == 'vgg16_bn':
        return vgg16_bn()
    elif classifier == 'vgg19_bn':
        return vgg19_bn()
    elif classifier == 'resnet18':
        return resnet18()
    elif classifier == 'resnet34':
        return resnet34()
    elif classifier == 'resnet50':
        return resnet50()
    elif classifier == 'densenet121':
        return densenet121()
    elif classifier == 'densenet161':
        return densenet161()
    elif classifier == 'densnet169':
        return densenet169()
    elif classifier == 'mobilenet_v2':
        return mobilenet_v2()
    elif classifier == 'googlenet':
        return googlenet()
    elif classifier == 'inception_v3':
        return inception_v3()

class CIFAR10_Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = get_classifier(hparams.classifier)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        
    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = torch.sum(torch.max(predictions, 1)[1] == labels.data).float() / images.size(0)
        return loss, accuracy
    
    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        logs = {'loss/train': loss, 'accuracy/train': accuracy}
        return {'loss': loss, 'log': logs}
        
    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        logs = {'loss/val': loss, 'accuracy/val': accuracy}
        return logs
                
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss/val'] for x in outputs]).mean()
        accuracy = torch.stack([x['accuracy/val'] for x in outputs]).mean()
        logs = {'loss/val': loss, 'accuracy/val': accuracy}
        return {'val_loss': loss, 'log': logs}
    
    def test_step(self, batch, batch_nb):
        _, accuracy = self.forward(batch)
        corrects = accuracy * batch[0].size(0)
        logs = {'corrects': corrects}
        return logs
    
    def test_epoch_end(self, outputs):
        corrects = torch.stack([x['corrects'] for x in outputs]).sum()
        test_dataset_length = len(self.test_dataloader().dataset)
        accuracy = round((corrects / test_dataset_length).item(), 2)
        return {'progress_bar': {'Accuracy': accuracy}}
        
    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate, 
                                        weight_decay=self.hparams.weight_decay, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.reduce_lr_per, gamma=0.1)
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform_train)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
    
    def val_dataloader(self):
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform_val)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
    
    def test_dataloader(self):
        transform_test = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform_test)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=4)
        return dataloader