import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

# Load the trained CNN model (the same model from train_neural_network.py) 
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


# Transform for the live image input (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
])

# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load("diabetic_retinopathy_model.pth"))  # Assuming model is saved as 'diabetic_retinopathy_model.pth'
model.eval()


# Function to make prediction
def predict(image):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    output = model(image)
    prediction = (output.squeeze() > 0.5).float()  # Predict class (0 or 1)
    return "DR" if prediction == 1 else "No_DR"


# Set up webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is captured correctly
    if not ret:
        print("Failed to grab frame")
        break

    # Make prediction on the captured frame
    prediction = predict(frame)

    # Display the resulting frame with prediction text
    cv2.putText(frame, f'Prediction: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the live webcam feed with the prediction label
    cv2.imshow('Live Diabetic Retinopathy Prediction', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
