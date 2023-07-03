import os
import zipfile

import pytorch_lightning as pl
import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from torchvision import transforms


class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def _get_transform(self, train):
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        return transform

    def download_weights():
        url = (
            "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
        )

        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)

        # Total size in Mebibyte
        total_size = int(r.headers.get("content-length", 0))
        block_size = 2 ** 20  # Mebibyte
        t = tqdm(total=total_size, unit="MiB", unit_scale=True)

        with open("state_dicts.zip", "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("Error, something went wrong")

        print("Download successful. Unzipping file...")
        path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
        directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
            print("Unzip file successful!")

    def train_dataloader(self):
        transform = self._get_transform(train=True)
        dataset = CIFAR10(root=self.data_dir, train=True, transform=transform, download=True)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        return dataloader

    def val_dataloader(self):
        transform = self._get_transform(train=False)
        dataset = CIFAR10(root=self.data_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        return dataloader

    def test_dataloader(self):
        transform = self._get_transform(train=False)
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return dataloader

    def test_dataloader(self):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test_dataset = CIFAR10(root=self.args.data_dir, train=False, download=True, transform=transform)
            test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers)
            return test_dataloader
