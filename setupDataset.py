import torch
import torchvision
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class ToScaleTensor:
    def __init__(self) -> None:
        pass
    def __call__(self, img: Tensor) -> Tensor:
        img = (2.0 / 255.0) * torch.tensor(np.array(img)).permute(2, 0, 1) - 1.0
        img = img.unsqueeze(dim=0)
        return img


class Dataset:
    def __init__(self, data: Dataset, batch_size: int):
        self.data = data
        self.length = len(data)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return self.length

    def __call__(self) -> Tensor:
        indices = torch.randint(0, self.length, (self.batch_size,)).int()
        img = torch.cat([self.data[i][0] for i in indices], dim=0)
        return img

    def __getitem__(self, idx) -> Tensor:
        indices = torch.arange(idx * self.batch_size, min((idx + 1) * self.batch_size, self.length)).int()
        img = torch.cat([self.data[i][0] for i in indices], dim=0)
        return img

def get_dataloaders_celeba(size, batch_size_train=64, batch_size_test=256):
    train_loader = Dataset(
        torchvision.datasets.CelebA(
            root="./files",
            split="train",
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    ToScaleTensor(),
                    torchvision.transforms.Resize((size[0], size[1])),
                ]
            ),
        ),
        batch_size=batch_size_train,
    )

    test_loader = Dataset(
        torchvision.datasets.CelebA(
            root="./files",
            transform=torchvision.transforms.Compose(
                [
                    ToScaleTensor(),
                    torchvision.transforms.Resize((size[0], size[1])),
                ]
            ),
            split="test",
            download=True,
        ),
        batch_size=batch_size_test,
    )

    return train_loader, test_loader

def get_dataloaders_cifar(batch_size_train=64, batch_size_test=256):
    train_loader = Dataset(
        torchvision.datasets.CIFAR10(
            root="./files",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    ToScaleTensor(),

                ]
            ),
        ),
        batch_size=batch_size_train,
    )

    test_loader = Dataset(
        torchvision.datasets.CIFAR10(
            root="./files",
            transform=torchvision.transforms.Compose(
                [
                    ToScaleTensor(),

                ]
            ),
            train=False,
            download=True,
        ),
        batch_size=batch_size_test,
    )

    return train_loader, test_loader

def get_dataloaders(ds_split, size, batch_size_train, batch_size_test):
    print(f"preparing dataset {ds_split} ...")
    if ds_split == "celeba":
        return get_dataloaders_celeba(size, batch_size_train, batch_size_test)
    elif ds_split == "cifar":
        print(f"NOTE: cifar doesn't have option {size}")
        return get_dataloaders_cifar(batch_size_train, batch_size_test)
    else:
        raise NotImplementedError("dataset not implemented")

if __name__ == "__main__":
    x,y = get_dataloaders_cifar()
    x,y = get_dataloaders_celeba((64,64))
    print("good to go")
