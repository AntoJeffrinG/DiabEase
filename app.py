from flask import Flask, abort, request, render_template, Response, jsonify
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
app = Flask(__name__, template_folder="templates/html", static_folder="static")

# Twilio credentials (⚠️ consider using .env in production)
account_sid = 'AC489474d56084f1233868886a91e9bd20'
auth_token = 'f7adfc48a45f6dda9a463b2425c026d6'
twilio_phone_number = '+12184928094'

# Load Nutrition Recommendation Data
df_nutrition = pd.read_csv('data/nutrition_recommendations_based_on_blood_sugar.csv')

# ---------- Utility Functions ----------

def is_valid_phone_number(phone_number):
    return re.match(r"^\+?[1-9]\d{1,14}$", phone_number)

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

def suggest_diet(blood_sugar, nutrition_type):
    filtered_df = df_nutrition[
        (df_nutrition['Nutrition Type'].str.lower() == nutrition_type.lower()) &
        (df_nutrition['Interval (mg/dL)'].apply(lambda x: int(x.split('-')[0]) <= blood_sugar <= int(x.split('-')[1])))
    ]
    if not filtered_df.empty:
        row = filtered_df.iloc[0]
        food_items = row["Food Items"].split(", ")
        suggested_foods = ", ".join(food_items[:4]) if len(food_items) >= 4 else ", ".join(food_items)
        return {
            "Daily Carbohydrates (g)": row["Daily Carbohydrates (g)"],
            "Daily Calories (kcal)": row["Daily Calories (kcal)"],
            "Suggested Food Items": suggested_foods,
            "Carbohydrates per Serving (g)": row["Carbohydrates per serving (g)"],
            "Calories per Serving (kcal)": row["Calories per serving (kcal)"]
        }
    return "No suitable diet found."

def send_sms_alert(name, sugar_level, phone_number):
    client = Client(account_sid, auth_token)
    body = f"GlucoAlert: {name}, abnormal glucose level detected: {sugar_level} mg/dL. Please take necessary action."
    try:
        client.messages.create(body=body, from_=twilio_phone_number, to=phone_number)
    except Exception as e:
        print(f"Error sending SMS: {e}")

def train_insulin_model():
    insulin_types = ['Rapid-acting', 'Short-acting', 'Intermediate-acting', 'Long-acting']
    data = {
        'Blood_Sugar_Level': np.random.uniform(70, 400, 1000),
        'Insulin_Type': np.random.choice(insulin_types, 1000),
        'Insulin_Dosage': np.random.uniform(2, 40, 1000)
    }
    df = pd.DataFrame(data)
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[['Insulin_Type']])
    insulin_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Insulin_Type']))
    X = pd.concat([df[['Blood_Sugar_Level']], insulin_df], axis=1)
    y = df['Insulin_Dosage']
    model = LinearRegression().fit(X, y)
    joblib.dump(model, 'models/insulin_dosage_model.pkl')
    joblib.dump(encoder, 'models/insulin_encoder.pkl')

if not os.path.exists('models/insulin_dosage_model.pkl'):
    train_insulin_model()

def load_model():
    model = joblib.load('models/insulin_dosage_model.pkl')
    encoder = joblib.load('models/insulin_encoder.pkl')
    return model, encoder

# ---------- CNN Setup ----------

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, 5),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(2, 5),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(2, 5)
        )
        self.fc_model = torch.nn.Sequential(
            torch.nn.Linear(256, 120),
            torch.nn.Tanh(),
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, 1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return torch.sigmoid(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = CNN().to(device)
cnn_model.load_state_dict(torch.load('models/diabetic_retinopathy_model.pth', map_location=device))
cnn_model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = None

def predict(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image)
    image = image.unsqueeze(0).to(device) # type: ignore
    output = cnn_model(image)
    return "Diabetic Retinopathy" if output.item() > 0.5 else "No Diabetic Retinopathy"

# ---------- Routes ----------

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/nutrition', methods=['POST'])
def nutrition():
    blood_sugar = request.form['blood_sugar']
    nutrition_type = request.form['nutrition_type']
    if not is_valid_number(blood_sugar, 40, 600):
        return render_template('nutrition.html', result="Invalid input.")
    result = suggest_diet(float(blood_sugar), nutrition_type)
    return render_template('diet_result.html', result=result)

@app.route('/insulin', methods=['POST'])
def insulin():
    blood_sugar = request.form['blood_sugar']
    insulin_type = request.form['insulin_type']
    if not is_valid_number(blood_sugar, 70, 400):
        return render_template('insulin.html', result="Invalid input.")
    try:
        model, encoder = load_model()
        df_input = pd.DataFrame({'Blood_Sugar_Level': [float(blood_sugar)], 'Insulin_Type': [insulin_type]})
        encoded = encoder.transform(df_input[['Insulin_Type']])
        input_final = pd.concat([df_input[['Blood_Sugar_Level']], pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Insulin_Type']))], axis=1)
        dosage = model.predict(input_final)[0]
        return render_template('insulin_result.html', result=round(dosage, 2))
    except Exception as e:
        return render_template('insulin_result.html', result=f"Prediction error: {e}")

@app.route('/blood_sugar_alert')
def blood_sugar_alert_form():
    return render_template('blood_sugar_alert.html')

@app.route('/send_alert', methods=['POST'])
def send_alert():
    name = request.form['name']
    phone = request.form['phone_number']
    low = request.form['low_range']
    high = request.form['high_range']
    current = request.form['current_sugar_level']

    if not (is_valid_phone_number(phone) and all(map(lambda x: is_valid_number(x, 40, 600), [low, high, current]))):
        return render_template('blood_sugar_alert.html', message="Invalid input.")

    if not phone.startswith('+'): phone = '+91' + phone
    current = int(current)
    if current < int(low) or current > int(high):
        send_sms_alert(name, current, phone)
        message = "Alert sent! Sugar level is abnormal."
    else:
        message = "Sugar level is normal."
    return render_template('blood_sugar_result.html', message=message)

@app.route('/retina')
def retina():
    return render_template('retina.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            return jsonify({'prediction': predict(frame)})
    return jsonify({'prediction': 'No Camera'})

@app.route('/stop_video_feed')
def stop_video_feed():
    global cap
    if cap:
        cap.release()
    cap = None
    return jsonify({'status': 'Camera Stopped'})

def generate_frames():
    global cap
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        label = predict(frame)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/<page>.html')
def render_custom_page(page):
    try:
        return render_template(f"{page}.html")
    except:
        abort(404)

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

# ---------- Run the app ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
