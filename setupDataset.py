import torch
import torchvision
import numpy as np
import matplotlib.pyplot
from torch import Tensor
from torch.utils.data import Dataset
import typing
from torch.nn.functional import interpolate


class ToScaleTensor:
    def __init__(self, size) -> None:
        self.size = size

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


def get_dataloaders(size, batch_size_train=64, batch_size_test=256):
    train_loader = Dataset(
        torchvision.datasets.CelebA(
            root="./files",
            split="train",
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    ToScaleTensor(size),
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
                    ToScaleTensor(size),
                    torchvision.transforms.Resize((size[0], size[1])),
                ]
            ),
            split="test",
            download=True,
        ),
        batch_size=batch_size_test,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    print("good to go")
