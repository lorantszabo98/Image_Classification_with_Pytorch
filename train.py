import torch
import torch.optim as optim
from created_models.simple_cnn_model import SimpleCNN
from utils.data_loader import get_data_loaders


def train(model, train_loader, num_epochs=5):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

if __name__ == "__main__":
    model = SimpleCNN()
    train_loader, _ = get_data_loaders()
    train(model, train_loader)
    torch.save(model.state_dict(), 'simple_cnn_model.pth')

