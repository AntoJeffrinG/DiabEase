from flask import Flask, abort, request, render_template, send_from_directory,Response, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from twilio.rest import Client
import os
import joblib
import re
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms

# Flask app initialization
app = Flask(__name__, template_folder="public/html", static_folder="public")

# Serving static files
@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('public/css', filename)

@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('public/images', filename)

@app.route("/")
def home():
    return render_template("home.html")

# Routes for rendering HTML pages
@app.route("/<page>.html")
def render_page(page):
    file_path = os.path.join("public/html", f"{page}.html")

    if os.path.exists(file_path): 
        return render_template(f"{page}.html")
    else:
        abort(404)

# Load environment variables for sensitive information
account_sid = 'AC489474d56084f1233868886a91e9bd20'
auth_token = 'f7adfc48a45f6dda9a463b2425c026d6'
twilio_phone_number = '+12184928094'

# Nutrition Recommendation Logic
df_nutrition = pd.read_csv('nutrition_recommendations_based_on_blood_sugar.csv')

# Helper functions for input validation
def is_valid_phone_number(phone_number):
    return re.match(r"^\+?[1-9]\d{1,14}$", phone_number)

def is_valid_blood_sugar(blood_sugar):
    return 40 <= blood_sugar <= 600

# Helper function to validate numeric input
def is_valid_number(input_value, min_val=None, max_val=None):
    try:
        number = float(input_value)
        if min_val is not None and number < min_val:
            return False
        if max_val is not None and number > max_val:
            return False
        return True
    except ValueError:
        return False

# Optimized diet suggestion
def suggest_diet(blood_sugar, nutrition_type):
    filtered_df = df_nutrition[
        (df_nutrition['Nutrition Type'].str.lower() == nutrition_type.lower()) &
        (df_nutrition['Interval (mg/dL)'].apply(lambda x: int(x.split('-')[0]) <= blood_sugar <= int(x.split('-')[1])))
    ]
    
    if not filtered_df.empty:
        row = filtered_df.iloc[0]
        food_items = row["Food Items"].split(", ")
        
        # Pick only the first 4 food items
        suggested_foods = ", ".join(food_items[:4]) if len(food_items) >= 4 else ", ".join(food_items)

        return {
            "Daily Carbohydrates (g)": row["Daily Carbohydrates (g)"],
            "Daily Calories (kcal)": row["Daily Calories (kcal)"],
            "Suggested Food Items": suggested_foods,
            "Carbohydrates per Serving (g)": row["Carbohydrates per serving (g)"],
            "Calories per Serving (kcal)": row["Calories per serving (kcal)"]
        }
    
    return "No suitable diet found."

# Insulin Dosage Model Training
num_samples = 1000
insulin_types = ['Rapid-acting', 'Short-acting', 'Intermediate-acting', 'Long-acting']
np.random.seed(42)
data = {
    'Blood_Sugar_Level': np.random.uniform(70, 400, num_samples),
    'Insulin_Type': np.random.choice(insulin_types, num_samples),
    'Insulin_Dosage': np.random.uniform(2, 40, num_samples)
}
df_insulin = pd.DataFrame(data)

encoder = OneHotEncoder(sparse_output=False)
insulin_encoded = encoder.fit_transform(df_insulin[['Insulin_Type']])
insulin_df = pd.DataFrame(insulin_encoded, columns=encoder.get_feature_names_out(['Insulin_Type']))

X = pd.concat([df_insulin[['Blood_Sugar_Level']], insulin_df], axis=1)
y = df_insulin['Insulin_Dosage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'insulin_dosage_model.pkl')

# Function to load the model
def load_model():
    return joblib.load('insulin_dosage_model.pkl')

# Function to send SMS alert
def send_sms_alert(name, sugar_level, phone_number):
    client = Client(account_sid, auth_token)

    message_body = (f"GlucoAlert: {name}  We've detected an unusual glucose level. Please check your reading and take the necessary steps as per your care instructions.")

    try:
        message = client.messages.create(
            body=message_body,
            from_=twilio_phone_number,
            to=phone_number
        )
        print(f"Alert sent to {phone_number} {name}")
    except Exception as e:
        print(f"Error sending SMS: {e}")

# Route for Nutrition Recommendation
@app.route('/nutrition', methods=['POST'])
def nutrition():
    blood_sugar = request.form['blood_sugar']
    nutrition_type = request.form['nutrition_type']

    # Validate numeric input
    if not is_valid_number(blood_sugar, 40, 600):
        return render_template('nutrition.html', result="Invalid blood sugar level. Please enter a number between 40 and 600.")

    blood_sugar = float(blood_sugar)
    diet_recommendation = suggest_diet(blood_sugar, nutrition_type)
    return render_template('diet_result.html', result=diet_recommendation)

# Route for Insulin Dosage Prediction
@app.route('/insulin', methods=['POST'])
def insulin():
    blood_sugar_level = request.form['blood_sugar']
    insulin_type = request.form['insulin_type']

    # Validate numeric input
    if not is_valid_number(blood_sugar_level, 70, 400):
        return render_template('insulin.html', result="Invalid blood sugar level. Please enter a number between 70 and 400.")

    try:
        blood_sugar_level = float(blood_sugar_level)
        new_data = {'Blood_Sugar_Level': [blood_sugar_level], 'Insulin_Type': [insulin_type]}
        new_input_df = pd.DataFrame(new_data)
        new_input_encoded = encoder.transform(new_input_df[['Insulin_Type']])
        new_input_final = pd.concat([new_input_df[['Blood_Sugar_Level']], pd.DataFrame(new_input_encoded, columns=encoder.get_feature_names_out(['Insulin_Type']))], axis=1)
        
        # Load the saved model and make predictions
        model = load_model()
        predicted_dosage = model.predict(new_input_final)
        predicted_dosage_rounded = round(predicted_dosage[0], 2)
        return render_template('insulin_result.html', result=predicted_dosage_rounded)
    except Exception as e:
        return render_template('insulin_result.html', result=f"Error in prediction: {str(e)}")

# New Route for Blood Sugar Alert System
@app.route('/blood_sugar_alert')
def blood_sugar_alert_form():
    return render_template('blood_sugar_alert.html')

@app.route('/send_alert', methods=['POST'])
def send_alert():
    name = request.form['name']
    phone_number = request.form['phone_number']
    low_range = request.form['low_range']
    high_range = request.form['high_range']
    current_sugar_level = request.form['current_sugar_level']

    # Validate inputs
    if not is_valid_phone_number(phone_number):
        return render_template('blood_sugar_alert.html', message="Invalid phone number format.")
    
    if not is_valid_number(low_range, 40, 600) or not is_valid_number(high_range, 40, 600):
        return render_template('blood_sugar_alert.html', message="Invalid blood sugar range. Please enter values between 40 and 600.")
    
    if not is_valid_number(current_sugar_level, 40, 600):
        return render_template('blood_sugar_alert.html', message="Invalid current blood sugar level. Please enter a value between 40 and 600.")
    
    # Convert to numeric types
    low_range = int(low_range)
    high_range = int(high_range)
    current_sugar_level = int(current_sugar_level)

    # If the phone number doesn't have a country code, default to +91 (India)
    if not phone_number.startswith('+'):
        phone_number = '+91' + phone_number

    # Check if sugar level is out of the given range
    if current_sugar_level < low_range or current_sugar_level > high_range:
        send_sms_alert(name, current_sugar_level, phone_number)
        message = "Blood sugar level is out of range. An alert SMS has been sent!"
    else:
        message = f"Blood sugar level is within the normal range ({low_range}-{high_range} mg/dL)."

    return render_template('blood_sugar_result.html', message=message)

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

    cap.release()

@app.route('/')
def index():
    return render_template('retina.html')

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