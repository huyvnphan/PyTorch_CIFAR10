import torch
import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from cifar10_module import CIFAR10_Module
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

def main(hparams):
    
    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    
    classifier = get_classifier(hparams.classifier)
    model = CIFAR10_Module(hparams, classifier)
    trainer = Trainer(default_save_path=os.path.join(os.getcwd(), 'trained_models', hparams.classifier),
                      gpus=[hparams.gpus], max_epochs=hparams.epochs,
                      early_stop_callback=False)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='/raid/data/pytorch_dataset/cifar10/')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--reduce_lr_per', type=int, default=50)
    args = parser.parse_args()
    main(args)
    
# Log
# python cifar10_main.py --learning_rate 1e-4 --classifier vgg11_bn --gpu 2
# python cifar10_main.py --learning_rate 1e-4 --classifier vgg13_bn --gpu 2
# python cifar10_main.py --learning_rate 1e-4 --classifier vgg16_bn --gpu 3
# python cifar10_main.py --learning_rate 1e-4 --classifier vgg19_bn --gpu 3

# python cifar10_main.py --classifier resnet18 --gpu 2
# python cifar10_main.py --classifier resnet34 --gpu 3