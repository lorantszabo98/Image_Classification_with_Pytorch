import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_data_loaders(batch_size=64, statistics=False):
    # define the transformation of the images
    # we convert the numpy array to Pytorch tensors
    # we normalize the tensor by subtracting the mean and dividing by the standard deviation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Data augmentation: Random horizontal flip
        transforms.RandomRotation(10),  # Data augmentation: Random rotation (degrees)
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Data augmentation: Random shifts
        transforms.RandomCrop(32, padding=4),  # Data augmentation: Random crop
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_train_without_augmentation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # download of the train and test dataset to the data directory and apply the defined transformation
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_dataset_without_augmentation = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_without_augmentation)

    # Print the number of images before and after data augmentation
    print(f"Number of images in the original training set: {len(train_dataset_without_augmentation)}")
    print(f"Number of images in the training set with data augmentation: {len(train_dataset)}")

    if statistics:
        print("\nStatistics about the train and test data\n")
        # Print the number of images in the training and test sets
        print(f"Number of images in the training set: {len(train_dataset)}")
        print(f"Number of images in the test set: {len(test_dataset)}\n")

        print("Class distribution in the train set:")
        class_distribution(train_dataset)
        print("\n")

        print("Class distribution in the test set:")
        class_distribution(test_dataset)
        print("\n")

    # define an iterable over a dataset, which we can use for training and testing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


def class_distribution(dataset):

    class_counts = torch.zeros(10, dtype=torch.long)
    for _, label in dataset:
        class_counts[label] += 1

    class_labels = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    for i, count in enumerate(class_counts):
        print(f"Class {class_labels[i]}: {count} images")
