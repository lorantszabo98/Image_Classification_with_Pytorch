import os
import torch
import torch.optim as optim
from created_models.simple_cnn_model import SimpleCNN
from utils.data_loader import get_data_loaders
from tqdm import tqdm


def train(model, train_loader, num_epochs=5):
    # define criterion and optimizer for training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # we iterate for the specified number of epochs
    for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
        running_loss = 0.0
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

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.3f}")

    print('Finished Training')

if __name__ == "__main__":
    model = SimpleCNN()

    train_loader, _ = get_data_loaders()
    train(model, train_loader)

    # create the saving directory
    save_directory = './trained_models'
    os.makedirs(save_directory, exist_ok=True)
    # get the model name for saving
    model_name = model.__class__.__name__
    # save the model
    # the state dictionary includes the learned parameters of the model
    torch.save(model.state_dict(), os.path.join(save_directory, f"{model_name}_model.pth"))


