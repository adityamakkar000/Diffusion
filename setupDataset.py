
import torch
import torchvision 
import matplotlib.pyplot

batch_size = 64
batch_size_test = 256
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
            batch_size=batch_size,
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




         
if __name__ == '__main__': 
    print('good to go :)') 
