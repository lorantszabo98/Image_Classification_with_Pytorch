import os
import torch
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from created_models.simple_cnn_model import SimpleCNN, SimpleCNN_v2, ImprovedCNN
from utils.data_loader import get_data_loaders
from utils.saving_loading_models import save_model
import torchvision.models as models
import torch.backends.cudnn as cudnn
# from torch.optim.lr_scheduler import StepLR


def plot_and_save_training_results(data, label, num_epochs, save_path):
    plt.plot(range(1, num_epochs + 1), data['train'], label='train')
    plt.plot(range(1, num_epochs + 1), data['val'], label='validation')
    plt.title(f'Training and validation {label}')
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.legend()

    plt.savefig(os.path.join(save_path, f"{label}.png"))
    plt.close()

    print(f"Training graph saved to {save_path}")


def train_val_step(dataloader, model, loss_function, optimizer, device):
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    running_loss = 0
    correct = 0
    total = 0

    for data in dataloader:
        image, labels = data
        image, labels = image.to(device), labels.to(device)
        outputs = model(image)
        loss = loss_function(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if optimizer is not None:
            optimizer.zero_grad()
            # perform backpropagation
            loss.backward()
            # update the model parameters
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader.dataset), correct / total


def train(model, train_loader, val_loader, device, num_epochs=5, mode='default'):

    save_directory = './training_graphs_and_logs'
    model_name = model.__class__.__name__

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
        model.fc = nn.Linear(num_ftrs, 10).to(device)

        optimizer = optim.SGD([
            {'params': model.fc.parameters()},
            {'params': model.layer3.parameters(), 'lr': 0.001}
            # {'params': model.layer4.parameters(), 'lr': 0.001}
        ], lr=0.001, momentum=0.9)

        save_path = os.path.join(save_directory, f"{model_name}_epochs_{num_epochs}_feature_extractor")

        # optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    elif mode == "fine_tuning":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10).to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        save_path = os.path.join(save_directory, f"{model_name}_epochs_{num_epochs}_fine_tuning")

    elif mode == "default":
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        save_path = os.path.join(save_directory, f"{model_name}_epochs_{num_epochs}")

    else:
        raise ValueError(f"Invalid mode. Supported values are: {', '.join(allowed_modes)}")

    accuracy_tracking = {'train': [], 'val': []}
    loss_tracking = {'train': [], 'val': []}
    best_loss = float('inf')

    os.makedirs(save_path, exist_ok=True)

    log_file_path = os.path.join(save_path, 'log.txt')
    log_file = open(log_file_path, 'a')

    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # we iterate for the specified number of epochs
    for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
        training_loss, training_accuracy = train_val_step(train_loader, model, criterion, optimizer, device)
        loss_tracking['train'].append(training_loss)
        accuracy_tracking['train'].append(training_accuracy)

        with torch.inference_mode():
            val_loss, val_accuracy = train_val_step(val_loader, model, criterion, None, device)
            loss_tracking['val'].append(val_loss)
            accuracy_tracking['val'].append(val_accuracy)
            if val_loss < best_loss:
                print('Saving best model')
                save_model('./trained_models', model, mode=mode)
                best_loss = val_loss

        print(f'Training accuracy: {training_accuracy:.6}, Validation accuracy: {val_accuracy:.6}')
        print(f'Training loss: {training_loss:.6}, Validation loss: {val_loss:.6}')

        # Append the information to the log file
        log_file.write(f"Epoch {epoch + 1}: "
                       f'Training accuracy: {training_accuracy:.6}, Validation accuracy: {val_accuracy:.6}, '
                       f'Training loss: {training_loss:.6}, Validation loss: {val_loss:.6}\n')

    print('\nFinished Training\n')

    plot_and_save_training_results(loss_tracking, 'loss', num_epochs, save_path)
    plot_and_save_training_results(accuracy_tracking, 'accuracy', num_epochs, save_path)

    log_file.close()


if __name__ == "__main__":
    model = SimpleCNN()
    # model = SimpleCNN_v2()
    # model = ImprovedCNN()
    # model = models.resnet18()

    # Feature extractor
    # model = models.resnet18(weights='IMAGENET1K_V1')
    model = models.resnet34(weights='IMAGENET1K_V1')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    train_loader, val_loader = get_data_loaders()
    train(model, train_loader, val_loader, device, num_epochs=3)


