import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from cifar10_module import CIFAR10_Module

def main(hparams):    
    torch.cuda.set_device(hparams.gpu)
    model = CIFAR10_Module(hparams)
    trainer = Trainer(gpus=[hparams.gpu])
    trainer.test(model)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='/raid/data/pytorch_dataset/cifar10/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--reduce_lr_per', type=int, default=50)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'])
    parser.add_argument('--pretrained', type=bool, default=True)
    args = parser.parse_args()
    main(args)