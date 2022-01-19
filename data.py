import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10


class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.data_dir = args.data_dir

        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=5),
                T.RandomRotation(10),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.data_dir, train=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.data_dir, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
