import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_data_loaders(batch_size=64):
    # define the transformation of the images
    # we convert the numpy array to Pytorch tensors
    # we normalize the tensor by subtracting the mean and dividing by the standard deviation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # download of the train and test dataset to the data directory and apply the defined transformation
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # define an iterable over a dataset, which we can use for training and testing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
