import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import threading
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Load the trained CNN model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2, stride=5),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2, stride=5)
        )
        self.fc_model = torch.nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=120),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=84, out_features=1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = torch.sigmoid(x)
        return x


# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load("diabetic_retinopathy_model.pth", map_location=device))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cap = None  # Global variable for video capture

def predict(image):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    prediction = (output.squeeze() > 0.5).float()
    return "Diabetic Retinopathy" if prediction == 1 else "No Diabetic Retinopathy"

def generate_frames():
    global cap
    cap = cv2.VideoCapture(0)  # Open webcam
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        prediction = predict(frame)
        cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Properly release the capture object after the loop ends
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    if cap is not None and cap.isOpened():
        success, frame = cap.read()
        if success:
            prediction = predict(frame)
            return jsonify({'prediction': prediction})
    return jsonify({'prediction': 'No Camera'})

@app.route('/stop_video_feed')
def stop_video_feed():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    return jsonify({'status': 'Camera Stopped'})

if __name__ == '__main__':
    app.run(debug=True)