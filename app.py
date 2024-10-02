from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from twilio.rest import Client

app = Flask(__name__)

# Nutrition Recommendation Logic
df_nutrition = pd.read_csv('nutrition_recommendations_based_on_blood_sugar.csv')

def suggest_diet(blood_sugar, nutrition_type):
    for i, row in df_nutrition.iterrows():
        interval_range = row["Interval (mg/dL)"].split('-')
        lower_bound = int(interval_range[0])
        upper_bound = int(interval_range[1])

        if lower_bound <= blood_sugar <= upper_bound and row["Nutrition Type"].lower() == nutrition_type.lower():
            return {
                "Daily Carbohydrates (g)": row["Daily Carbohydrates (g)"],
                "Daily Calories (kcal)": row["Daily Calories (kcal)"],
                "Suggested Food Items": row["Food Items"],
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

# Twilio account credentials
account_sid = 'AC1b5e6188bcd73beaec18333df0f4cc1c'  # Replace with your Twilio Account SID
auth_token = '37dc020e293aa82a5dd8a3a896611f33'    # Replace with your Twilio Auth Token
twilio_phone_number = '+13522928291'  # Replace with your Twilio Phone Number

# Function to send SMS alert
def send_sms_alert(name, sugar_level, phone_number):
    client = Client(account_sid, auth_token)

    message_body = (f"Alert: {name}, your blood sugar level is {sugar_level} mg/dL, "
                    "which is out of your specified range. Please take action.")

    try:
        message = client.messages.create(
            body=message_body,
            from_=twilio_phone_number,
            to=phone_number
        )
        print(f"Alert sent to {phone_number}. Message SID: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/model1')
def model1():
    return render_template('model1.html')

@app.route('/model2')
def model2():
    return render_template('model2.html')

@app.route('/nutrition', methods=['POST'])
def nutrition():
    blood_sugar = float(request.form['blood_sugar'])
    nutrition_type = request.form['nutrition_type']
    diet_recommendation = suggest_diet(blood_sugar, nutrition_type)
    return render_template('nutrition.html', result=diet_recommendation)

@app.route('/insulin', methods=['POST'])
def insulin():
    blood_sugar_level = float(request.form['blood_sugar'])
    insulin_type = request.form['insulin_type']
    new_data = {'Blood_Sugar_Level': [blood_sugar_level], 'Insulin_Type': [insulin_type]}
    new_input_df = pd.DataFrame(new_data)
    new_input_encoded = encoder.transform(new_input_df[['Insulin_Type']])
    new_input_final = pd.concat([new_input_df[['Blood_Sugar_Level']], pd.DataFrame(new_input_encoded, columns=encoder.get_feature_names_out(['Insulin_Type']))], axis=1)
    predicted_dosage = model.predict(new_input_final)
    predicted_dosage_rounded = round(predicted_dosage[0], 2)
    return render_template('insulin.html', result=predicted_dosage_rounded)

# New Route for Blood Sugar Alert System
@app.route('/blood_sugar_alert')
def blood_sugar_alert_form():
    return render_template('blood_sugar_alert.html')

@app.route('/send_alert', methods=['POST'])
def send_alert():
    name = request.form['name']
    phone_number = request.form['phone_number']
    low_range = int(request.form['low_range'])
    high_range = int(request.form['high_range'])
    current_sugar_level = int(request.form['current_sugar_level'])

    if not phone_number.startswith('+'):
        phone_number = '+91' + phone_number  # Default to Indian number

    if current_sugar_level < low_range or current_sugar_level > high_range:
        send_sms_alert(name, current_sugar_level, phone_number)
        message = "Blood sugar level is out of range. An alert SMS has been sent!"
    else:
        message = f"Blood sugar level is within the normal range ({low_range}-{high_range} mg/dL)."

    return render_template('blood_sugar_result.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
