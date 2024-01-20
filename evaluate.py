import torch
from created_models.simple_cnn_model import SimpleCNN
from utils.data_loader import get_data_loaders
import os


def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: {:.2f}%'.format(accuracy))

if __name__ == "__main__":
    model = SimpleCNN()

    model_name = model.__class__.__name__
    save_directory = './trained_models'

    model.load_state_dict(torch.load(os.path.join(save_directory, f"{model_name}_model.pth")))
    _, test_loader = get_data_loaders()
    evaluate(model, test_loader)
