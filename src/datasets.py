from torchvision import datasets, transforms
import torch

def get_dataset(dataset, batch_size):
    if dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./mnist_train', train=True, transform=transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
        image_size, channel_size = 28, 1
    elif dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(root='./fashion_data', train=True, transform=transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
        image_size, channel_size = 28, 1
    else:
        pass

    return train_loader, (image_size, channel_size)