import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.preact_resnet import PreActResNet18
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from data import CIFAR10Data
from module import CIFAR10Module

all_classifiers = {
    "vgg11_bn": vgg11_bn,
    "vgg13_bn": vgg13_bn,
    "vgg16_bn": vgg16_bn,
    "vgg19_bn": vgg19_bn,
    "resnet18": resnet18,
    "preact_resnet18": PreActResNet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "mobilenet_v2": mobilenet_v2,
    "googlenet": googlenet,
    "inception_v3": inception_v3,
}


def main(args):
    seed_everything(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.logger == "wandb":
        logger = WandbLogger(
            name=args.description,
            project="PyTorch_CIFAR10",
            log_model=False,
            save_dir="logs",
        )
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger("cifar10", name=args.classifier)

    checkpoint_callback = ModelCheckpoint(
        monitor="accuracy/val", mode="max", save_last=True
    )

    # Resume ckpt in logs
    if args.resume_path != "None":
        path = args.resume_path
    else:
        path = None

    # Prepare trainer
    trainer = Trainer(
        logger=logger,
        gpus=-1,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        resume_from_checkpoint=path,
        precision=args.precision,
        callbacks=[checkpoint_callback],
    )

    model = all_classifiers[args.classifier](pretrained=bool(args.pretrained))
    module = CIFAR10Module(model, args)
    data = CIFAR10Data(args)

    if bool(args.test_phase):
        trainer.test(module, data.test_dataloader())
    else:
        trainer.fit(module, data)
        trainer.test()
        # Save final weights
        file_name = "cifar10_models/state_dicts/" + args.description + ".pt"
        torch.save(module.model.state_dict(), file_name)


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--description", type=str, default="debug_run")
    parser.add_argument("--data_dir", type=str, default="/home/huy/data/cifar10")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])

    # TRAINER args
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32])
    parser.add_argument("--resume_path", type=str, default="None")
    parser.add_argument("--resume_id", type=str, default="None")
    parser.add_argument(
        "--logger", type=str, default="wandb", choices=["tensorboard", "wandb"]
    )

    # HYPER-PARAMS args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-3)

    args = parser.parse_args()
    main(args)
