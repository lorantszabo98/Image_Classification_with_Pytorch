import torch
import os
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import random
from torch import nn
from created_models.simple_cnn_model import SimpleCNN, SimpleCNN_v2, ImprovedCNN
from utils.data_loader import get_data_loaders
from utils.saving_loading_models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def display_random_predictions(model, test_loader, class_labels, num_images=8, mode='default'):
    # load the model based on the mode
    if mode == "feature_extractor" or mode == "fine_tuning":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

        load_model('./trained_models', model, mode=mode)
    else:
        load_model('./trained_models', model)
    # set model to evaluation mode
    model.eval()

    # get the total number of images in the test set
    total_images = len(test_loader.dataset)

    # randomly select 6 indices from the test set
    selected_indices = random.sample(range(total_images), num_images)

    images_so_far = 0
    # create a set of subplots in a grid, the first param is number of rows, the second param is the number of columns
    fig, axs = plt.subplots(num_images // 2, 2, figsize=(10, 10))

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # images is a tensor with shape (batch_size, channels, height, width)
            # images.size()[0] is the number of images in the batch
            for j in range(images.size()[0]):
                # exit the loop when the desired number of images is reached
                if images_so_far >= num_images:
                    break

                # Check if the current index is in the randomly selected indices
                if i * test_loader.batch_size + j in selected_indices:
                    #  accesses a specific subplot in the grid of subplots
                    ax = axs[images_so_far // 2, images_so_far % 2]
                    # turns of the axis
                    ax.axis('off')
                    #  sets the title of the subplot
                    ax.set_title(f'predicted: {class_labels[preds[j]]}')

                    # convert the image tensor to NumPy array and transpose
                    img = np.transpose(images.cpu().data[j].numpy(), (1, 2, 0))
                    # unnormalize if normalization was applied during data loading
                    img = img * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])

                    # show the image in the subplot
                    ax.imshow(img)
                    # counting the processed images
                    images_so_far += 1

    # after all show the plot
    plt.show()


def evaluate(model, test_loader, mode='default'):

    if mode == "feature_extractor" or mode == "fine_tuning":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

        load_model('./trained_models', model, mode=mode)
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

    print("Evaluation metrics:\n")

    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    print('Precision: {:.2f}'.format(precision))
    print('Recall: {:.2f}'.format(recall))
    print('F1 Score: {:.2f}'.format(f1))

    # print out the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    summary(model=model,
            input_size=(64, 3, 32, 32),  # (batch_size, color_channels, height, width)
            col_names=["input_size", "output_size", "num_params"],
            col_width=20,
            row_settings=["var_names"]
    )

    # print out the model structure
    # print(model.eval())

    return y_true, y_pred

if __name__ == "__main__":
    # model = SimpleCNN()
    # model = SimpleCNN_v2()
    # model = ImprovedCNN()
    model = models.resnet18()
    # model = models.resnet34()

    class_labels = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    _, test_loader = get_data_loaders(statistics=False)
    true_labels, predictions = evaluate(model, test_loader, mode="fine_tuning")

    # Create classification report
    report = classification_report(true_labels, predictions)
    print('Classification Report:')
    print(report)

    # Create the confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Display the 8 random predictions from the testing dataset
    # display_random_predictions(model, test_loader, class_labels)