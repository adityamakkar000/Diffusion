
import torch
import torchvision
import matplotlib.pyplot


def get_dataloaders(batch_size_train=64, batch_size_test=256):

    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CelebA(
                root="./files",
                split='train',
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        ]
            ),
                ),
                batch_size=batch_size_train,
                shuffle=True

            )

    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CelebA(
                root="./files",
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        ]
                ),
                split='test',
                download=True,
                ),
            batch_size=batch_size_test,
            shuffle=True,
    )


    return train_loader, test_loader





if __name__ == '__main__':
    print('good to go :)')
