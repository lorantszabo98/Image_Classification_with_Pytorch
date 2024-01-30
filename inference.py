import torch
import time
from torchvision import transforms
from PIL import Image
from created_models.simple_cnn_model import ImprovedCNN

start_time = time.time()

# class labels for the CIFAR-10 dataset
class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Init the model, load the model's state dictionary and set the model to evaluation mode
model = ImprovedCNN()
model.load_state_dict(torch.load('trained_models/ImprovedCNN_model.pth'))
model.eval()

# Specify the path to the test image
image_path = 'test_images/automoible_004.jpg'
# Define the transformation to apply to the test image
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Apply the preprocessing
input_image = Image.open(image_path)
input_data = preprocess(input_image)
# Add a batch dimension to the 0 index of the shape of the input data to produce shape like this: (1, 3, 32, 32)
# before this line the shape was (3, 32, 32)
input_data = input_data.unsqueeze(0)

# Make predictions without gradient computation
with torch.no_grad():
    output = model(input_data)

# Find the class with the highest probability, 1 stand for that we are searching for the class, where this value appears
_, predicted_class = torch.max(output, 1)
# Get the predicted class label
predicted_label = class_labels[predicted_class.item()]

# Adding softmax to produce probabilities from the model output
probabilities = torch.nn.functional.softmax(output, dim=1)
# Convert probabilities to percentages
probabilities_percentage = (probabilities.squeeze().numpy() * 100).round(2)

end_time = time.time()
elapsed_time_ms = (end_time - start_time) * 1000

# Print out the results
print(f"Predicted Class: {predicted_label}")
print(f"Class Probabilities: {dict(zip(class_labels, probabilities_percentage))}")
print(f"Inference time: {elapsed_time_ms:.2f} ms")
