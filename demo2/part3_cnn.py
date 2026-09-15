
import torch.nn as nn
import numpy as np
import torch
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import classification_report

# COMP3710 Demo 2 - Part 3.1: CNN Classifier
# Step 1: Prepare the LFW images as 4D tensors for convolution.


# [Provided] Load the same LFW dataset used in Part 2.
lfw_people = fetch_lfw_people(
    min_faces_per_person=70,
    resize=0.4
)


# [Provided] CNNs use the original 2D images instead of flattened PCA vectors.
X = lfw_people.images
y = lfw_people.target

n_classes = len(lfw_people.target_names)


# [Provided] Check the input value range.
print("X min:", X.min())
print("X max:", X.max())


# [Provided] Create the same 75/25 train-test split as Part 2.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# ★ CORE: Add the channel dimension required by PyTorch Conv2d.
# Before: (samples, height, width)
# After:  (samples, channels, height, width)
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]


print("\n--- CNN Input Shapes ---")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
print("Classes:", n_classes)

# Q2 | Official: build a CNN with two 3x3 convolution layers,
# 32 filters each, followed by dense layers for classification.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceCNN(nn.Module):
    """CNN classifier for the 50x37 grayscale LFW face images.

    Input:
        x: tensor with shape (batch_size, 1, 50, 37)

    Output:
        logits: tensor with shape (batch_size, 7)
    """

    def __init__(self, n_classes):
        super().__init__()

        # ★ CORE: first required 3x3 convolution with 32 filters.
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # 2x2 max pooling reduces height and width by approximately half.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ★ CORE: second required 3x3 convolution with 32 filters.
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # ReLU provides the non-linear activation needed between layers.
        self.relu = nn.ReLU()

        # After two pooling operations:
        # 50x37 -> 25x18 -> 12x9
        self.flattened_features = 32 * 12 * 9

        # Dense layer learns combinations of the extracted CNN features.
        self.fc1 = nn.Linear(self.flattened_features, 128)

        # ★ CORE: final dense layer produces one score for each of the 7 people.
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # First convolution block: 1x50x37 -> 32x25x18.
        x = self.pool(self.relu(self.conv1(x)))

        # Second convolution block: 32x25x18 -> 32x12x9.
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten spatial feature maps for the dense classifier.
        x = torch.flatten(x, start_dim=1)

        # Hidden dense layer.
        x = self.relu(self.fc1(x))

        # ★ CORE: return raw class scores (logits).
        x = self.fc2(x)

        return x


# Create the CNN and move it to GPU when CUDA is available.
model = FaceCNN(n_classes=n_classes).to(device)

print("\n--- CNN Model ---")
print("Device:", device)
print(model)

# Q3 | Official: train the CNN classifier using the LFW training data.

# Convert NumPy arrays into PyTorch tensors.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Combine images and labels into one training dataset.
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# Create mini-batches for training.
batch_size = 32
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# CrossEntropyLoss compares the 7 output logits with the true integer class labels.
criterion = nn.CrossEntropyLoss()

# Adam updates all trainable CNN parameters using the calculated gradients.
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Number of complete passes through the training set.
num_epochs = 20


print("\n--- CNN Training ---")

for epoch in range(num_epochs):

    # Put the model into training mode.
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # Move the current mini-batch to the selected device.
        images = images.to(device)
        labels = labels.to(device)

        # ★ CORE: clear gradients left from the previous mini-batch.
        optimizer.zero_grad()

        # ★ CORE: forward pass produces one set of 7 class logits per image.
        outputs = model(images)

        # ★ CORE: calculate classification loss.
        loss = criterion(outputs, labels)

        # ★ CORE: backpropagation computes gradients for every trainable parameter.
        loss.backward()

        # ★ CORE: Adam uses those gradients to update the CNN parameters.
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Select the class with the largest output logit.
        predicted = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    print(
        f"Epoch {epoch + 1:02d}/{num_epochs} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Train Accuracy: {epoch_accuracy:.4f}"
    )

    # Q4 | Evaluate the trained CNN on the unseen LFW test set.

# Convert the test images to a PyTorch tensor.
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# Put the model into evaluation mode.
model.eval()

# Disable gradient calculation because evaluation does not update model parameters.
with torch.no_grad():

    # ★ CORE: predict class logits for all unseen test faces.
    test_outputs = model(X_test_tensor)

    # ★ CORE: choose the class with the highest logit for each face.
    test_predictions = test_outputs.argmax(dim=1)


# Move predictions back to CPU/NumPy for sklearn evaluation.
test_predictions = test_predictions.cpu().numpy()

# Calculate overall test accuracy.
test_correct = np.sum(test_predictions == y_test)
test_accuracy = test_correct / len(y_test)

print("\n--- CNN Test Performance ---")
print("Total testing:", len(y_test))
print("Total correct:", test_correct)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=lfw_people.target_names,
        zero_division=0
    )
)