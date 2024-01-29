import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from created_models.simple_cnn_model import SimpleCNN, SimpleCNN_v2, ImprovedCNN
from utils.data_loader import get_data_loaders


def plot_and_save_training_results(epochs, train_accuracies, train_losses, model_name, num_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title('Training Accuracy and Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    save_directory = './training_graphs'
    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, f"{model_name}_epochs_{num_epochs}.png")

    plt.savefig(save_path)
    plt.close()

    print(f"Training graph saved to {save_path}")


def train(model, train_loader, num_epochs=5):
    # define criterion and optimizer for training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_accuracies = []
    train_losses = []

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

    print('Finished Training')

    model_name = model.__class__.__name__
    # Call the plotting function
    plot_and_save_training_results(range(1, num_epochs + 1), train_accuracies, train_losses, model_name, num_epochs)

if __name__ == "__main__":
    model = SimpleCNN()
    # model = SimpleCNN_v2()
    # model = ImprovedCNN()

    train_loader, _ = get_data_loaders()
    train(model, train_loader, num_epochs=10)

    # create the saving directory
    save_directory = './trained_models'
    os.makedirs(save_directory, exist_ok=True)
    # get the model name for saving
    model_name = model.__class__.__name__

    # save the model
    # the state dictionary includes the learned parameters of the model
    torch.save(model.state_dict(), os.path.join(save_directory, f"{model_name}_model.pth"))


