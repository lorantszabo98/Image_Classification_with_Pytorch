import os
import torch
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from created_models.simple_cnn_model import SimpleCNN, SimpleCNN_v2, ImprovedCNN
from utils.data_loader import get_data_loaders
from   utils.saving_loading_models import save_model
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR


cudnn.benchmark = True


def plot_and_save_training_results(epochs, train_accuracies, train_losses, model_name, num_epochs, mode='default'):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title('Training Accuracy and Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    save_directory = './training_graphs'
    os.makedirs(save_directory, exist_ok=True)
    if mode == "feature_extractor":
        save_path = os.path.join(save_directory, f"{model_name}_epochs_{num_epochs}_feature_extractor.png")
    elif mode == "fine_tuning":
        save_path = os.path.join(save_directory, f"{model_name}_epochs_{num_epochs}_fine_tuned.png")
    else:
        save_path = os.path.join(save_directory, f"{model_name}_epochs_{num_epochs}.png")

    plt.savefig(save_path)
    plt.close()

    print(f"Training graph saved to {save_path}")


def train(model, train_loader, num_epochs=5, mode='default'):

    # define criterion and optimizer for training
    criterion = torch.nn.CrossEntropyLoss()
    allowed_modes = {'default', 'fine_tuning', 'feature_extractor'}
    # if we use a model in feature extractor mode, we freeze every parameter
    # then we add a new layer and only this layer will be trained
    if mode == "feature_extractor":
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer3.parameters():
            param.requires_grad = True

        # for param in model.layer4.parameters():
        #     param.requires_grad = True

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

        optimizer = optim.SGD([
            {'params': model.fc.parameters()},
            {'params': model.layer3.parameters(), 'lr': 0.001}
            # {'params': model.layer4.parameters(), 'lr': 0.001}
        ], lr=0.001, momentum=0.9)

        # optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    elif mode == "fine_tuning":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif mode == "default":
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        raise ValueError(f"Invalid mode. Supported values are: {', '.join(allowed_modes)}")

    train_accuracies = []
    train_losses = []

    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # we iterate for the specified number of epochs
    for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # unpack data obtained from a data loader, which is a tuple
            # the first element is the input data and the second element is the corresponding labels
            inputs, labels = data
            # zero the gradients
            optimizer.zero_grad()
            # calculate the predictions
            outputs = model(inputs)
            # compute the loss
            loss = criterion(outputs, labels)
            # perform backpropagation
            loss.backward()
            # update the model parameters
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy_train = correct_train / total_train

        train_accuracies.append(accuracy_train)
        train_losses.append(epoch_loss)

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.3f}, Accuracy: {accuracy_train * 100:.2f}%")

        # scheduler.step()

    print('Finished Training')

    model_name = model.__class__.__name__
    # Call the plotting function
    plot_and_save_training_results(range(1, num_epochs + 1), train_accuracies, train_losses, model_name, num_epochs, mode=mode)

    # saving the model
    save_model('./trained_models', model, mode=mode)


if __name__ == "__main__":
    # model = SimpleCNN()
    # model = SimpleCNN_v2()
    # model = ImprovedCNN()
    # model = models.resnet18()

    # Feature extractor
    model = models.resnet18(weights='IMAGENET1K_V1')
    # model = models.resnet34(weights='IMAGENET1K_V1')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    train_loader, _ = get_data_loaders()
    train(model, train_loader, num_epochs=10, mode='feature_extractor')


