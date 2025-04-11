import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# CNN Model Definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = torch.sigmoid(x)
        return x


# Custom Dataset Class for Diabetic Retinopathy
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Loop over 'DR' and 'No_DR' directories in the given root_dir
        for label, category in enumerate(['DR', 'No_DR']):
            category_dir = os.path.join(self.root_dir, category)
            for img_name in os.listdir(category_dir):
                img_path = os.path.join(category_dir, img_name)
                if img_path.endswith('.jpg') or img_path.endswith('.png'):  # Filter by image extension
                    self.images.append(img_path)
                    self.labels.append(label)

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        return {'image': img, 'label': label}


# Define the necessary transformations (resize, normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the image to 128x128
    transforms.ToTensor(),  # Convert the image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
])

# Create datasets for training, validation, and testing
train_dataset = DiabeticRetinopathyDataset(root_dir='data/Diagnosis of Diabetic Retinopathy/train', transform=transform)
valid_dataset = DiabeticRetinopathyDataset(root_dir='data/Diagnosis of Diabetic Retinopathy/valid', transform=transform)
test_dataset = DiabeticRetinopathyDataset(root_dir='data/Diagnosis of Diabetic Retinopathy/test', transform=transform)

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Training loop with validation
def train_and_validate(model, train_loader, valid_loader, optimizer, criterion, epochs=10000):
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Convert labels to float before passing them to the loss function
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Print training loss for this epoch
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {np.mean(train_losses)}")

        # Validation step
        model.eval()
        valid_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in valid_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())  # Convert labels to float
                valid_losses.append(loss.item())

                predicted = (outputs.squeeze() > 0.5).float()  # Convert to binary prediction
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        valid_accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {np.mean(valid_losses)}, Validation Accuracy: {valid_accuracy:.4f}")


# Test the model and plot confusion matrix
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    outputs = []
    labels_list = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs_batch = model(images)
            predicted = (outputs_batch.squeeze() > 0.5).float()

            outputs.append(predicted.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    # Optional: Confusion matrix and other evaluation metrics
    outputs = np.concatenate(outputs, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    cm = confusion_matrix(labels_list, outputs)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['No_DR', 'DR'], yticklabels=['No_DR', 'DR'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Initialize model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.BCELoss()

# Train the model with validation
train_and_validate(model, train_loader, valid_loader, optimizer, criterion, epochs=10000)

# Evaluate the model on the test set
evaluate(model, test_loader)

# Save the trained model weights after training
torch.save(model.state_dict(), "diabetic_retinopathy_model.pth")
print("Model weights saved successfully!")
