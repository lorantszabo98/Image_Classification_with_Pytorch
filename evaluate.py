import torch
import os
import torchvision.models as models
from torch import nn
from created_models.simple_cnn_model import SimpleCNN, SimpleCNN_v2, ImprovedCNN
from utils.data_loader import get_data_loaders
from utils.saving_loading_models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter


def evaluate(model, test_loader, feature_extractor=False):

    if feature_extractor:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

        load_model('./trained_models', model, feature_extractor_mode=True)
    else:
        load_model('./trained_models', model)
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    print('Precision: {:.2f}'.format(precision))
    print('Recall: {:.2f}'.format(recall))
    print('F1 Score: {:.2f}'.format(f1))

    # print out the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # print out the model structure
    # print(model.eval())

if __name__ == "__main__":
    # model = SimpleCNN()
    # model = SimpleCNN_v2()
    # model = ImprovedCNN()
    model = models.resnet18()

    # Create a SummaryWriter to use TensorBoard
    writer = SummaryWriter('logs')
    # Add the model graph to TensorBoard
    dummy_input = torch.randn(1, 3, 32, 32)
    writer.add_graph(model, dummy_input)
    # Close the SummaryWriter
    writer.flush()
    writer.close()

    _, test_loader = get_data_loaders()
    evaluate(model, test_loader, feature_extractor=False)
