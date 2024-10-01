from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

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

if __name__ == '__main__':
    app.run(debug=True)