import os
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torchvision.datasets import CIFAR10
import sys
sys.path.append('/content/PyTorch_CIFAR10/data.py')
from data import CIFAR10Data
from module import CIFAR10Module
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main(args):

    if bool(args.download_weights):
        CIFAR10Data.download_weights()
    else:
        seed_everything(0)

        if args.logger == "wandb":
            logger = WandbLogger(name=args.classifier, project="cifar10")
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger("cifar10", name=args.classifier)

        checkpoint_callback = ModelCheckpoint(monitor="acc/val", mode="max", save_last=False)

        trainer = Trainer(
            fast_dev_run=bool(args.dev),
            logger=logger if not bool(args.dev + args.test_phase) else None,
            accelerator="gpu",
            deterministic=True,
            log_every_n_steps=1,
            max_epochs=args.max_epochs,
            precision=args.precision,
        )

        model = CIFAR10Module(args)
        data = CIFAR10Data(args)

        if bool(args.pretrained):
            state_dict = os.path.join(
                "cifar10_models", "state_dicts", args.classifier + ".pt"
            )
            model.model.load_state_dict(torch.load(state_dict))

        if bool(args.test_phase):
            trainer.test(model, dataloaders=data.test_dataloader())
        else:
            trainer.fit(model, datamodule=data)
            trainer.test(dataloaders=data.test_dataloader())

        # Manually save the best checkpoint
        if trainer.global_rank == 0:
            best_checkpoint_path = checkpoint_callback.best_model_path
            if best_checkpoint_path is not None:
                best_checkpoint_dir = os.path.join(args.data_dir, args.classifier, "checkpoints")
                os.makedirs(best_checkpoint_dir, exist_ok=True)
                best_checkpoint_file = os.path.join(best_checkpoint_dir, "best_model.pt")
                torch.save(model.state_dict(), best_checkpoint_file)
                print(f"Saved best checkpoint: {best_checkpoint_file}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/data/huy/cifar10")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)

    # CIFAR10Module args
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_step_size", type=int, default=50)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
