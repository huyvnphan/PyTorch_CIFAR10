import os
import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from cifar10_module import CIFAR10_Module

def main(hparams):
    
    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    
    # Set GPU
    torch.cuda.set_device(hparams.gpu)
    
    # Train
    classifier = CIFAR10_Module(hparams)
    trainer = Trainer(default_save_path=os.path.join(os.getcwd(), 'tensorboard_logs', hparams.classifier),
                      gpus=[hparams.gpu], max_epochs=hparams.max_epochs,
                      early_stop_callback=False)
    trainer.fit(classifier)
    
    # Save weights from checkpoint
    checkpoint_path = os.path.join(os.getcwd(), 'tensorboard_logs', hparams.classifier, 'lightning_logs', 'version_0', 'checkpoints')
    classifier = CIFAR10_Module.load_from_checkpoint(os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]))
    statedict_path = os.path.join(os.getcwd(), 'cifar10_models', 'state_dicts', hparams.classifier + '.pt')
    torch.save(classifier.model.state_dict(), statedict_path)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='/raid/data/pytorch_dataset/cifar10/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=2e-2)
    parser.add_argument('--reduce_lr_per', type=int, default=30)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'])
    parser.add_argument('--pretrained', type=bool, default=False)
    args = parser.parse_args()
    main(args)