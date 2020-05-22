import os, shutil
import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from cifar10_module import CIFAR10_Module

def main(hparams):    
    # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
    if type(hparams.gpus) == str:
        if len(hparams.gpus) == 2: # GPU number and comma e.g. '0,' or '1,'
            torch.cuda.set_device(int(hparams.gpus[0]))
            
    model = CIFAR10_Module(hparams, pretrained=True)
    trainer = Trainer(gpus=hparams.gpus, default_save_path=os.path.join(os.getcwd(), 'test_temp'))
    trainer.test(model)
    shutil.rmtree(os.path.join(os.getcwd(), 'test_temp'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='/data/huy/cifar10/')
    parser.add_argument('--gpus', default='0,')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    args = parser.parse_args()
    main(args)