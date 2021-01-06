import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from cifar10_data import CIFAR10Data
from cifar10_module import CIFAR10Module


def main(args):

    seed_everything(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    logger = WandbLogger(name=args.description, project="pytorch_cifar10")
    checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=False)

    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        logger=logger if not bool(args.dev) else None,
        gpus=-1,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        checkpoint_callback=checkpoint,
        precision=args.precision,
    )

    if bool(args.download_data):
        CIFAR10Data.download()

    model = CIFAR10Module(args)
    data = CIFAR10Data(args)

    if args.phase == "train":
        trainer.fit(model, data)
        trainer.test()
    else:
        trainer.test(model, data.test_dataloader())


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--description", type=str, default="default")
    parser.add_argument("--data_dir", type=str, default="/data/huy/cifar10")
    parser.add_argument("--download_data", type=int, default=0, choices=[0, 1])
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")

    parser.add_argument("--precision", type=int, default=16, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="3")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    args = parser.parse_args()
    main(args)
