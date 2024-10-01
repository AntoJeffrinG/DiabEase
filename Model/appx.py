import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

print("1 : get Nutrition Recomendation based on blood sugar level.")
print("2 : get insulin dosage recomendation based on blood sugar level.")

c = int(input("Enter your choice : "))

if c == 1:
    # Load the dataset
    df = pd.read_csv('nutrition_recommendations_based_on_blood_sugar.csv')

    # Function to suggest diet based on blood sugar level and nutrition type
    def suggest_diet(blood_sugar, nutrition_type):
        # Find the relevant interval
        for i, row in df.iterrows():
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
    
        return "No suitable diet found for the given input."

    # Main function to get user input and provide recommendations
    def main():
        try:
            blood_sugar = float(input("Enter your blood sugar level (mg/dL): "))
            nutrition_type = input("Enter your dietary preference (Vegan/Non-Vegan): ").strip()

            diet_recommendation = suggest_diet(blood_sugar, nutrition_type)

            if isinstance(diet_recommendation, dict):
                print("\nDiet Recommendation:")
                print("Daily Carbohydrates (g):", diet_recommendation["Daily Carbohydrates (g)"])
                print("Daily Calories (kcal):", diet_recommendation["Daily Calories (kcal)"])
                print("Suggested Food Items:", diet_recommendation["Suggested Food Items"])
                print("Carbohydrates per Serving (g):", diet_recommendation["Carbohydrates per Serving (g)"])
                print("Calories per Serving (kcal):", diet_recommendation["Calories per Serving (kcal)"])
            else:
                print(diet_recommendation)
        except ValueError:
            print("Invalid input. Please enter numerical values for blood sugar levels.")

    # Run the main function
    if __name__ == "__main__":
        main()
elif c == 2:
    # Step 1: Generate the Dataset
    num_samples = 1000
    insulin_types = ['Rapid-acting', 'Short-acting', 'Intermediate-acting', 'Long-acting']

    # Generating synthetic data
    np.random.seed(42)
    data = {
        'Blood_Sugar_Level': np.random.uniform(70, 400, num_samples),  # Blood sugar levels between 70 and 400 mg/dL
        'Insulin_Type': np.random.choice(insulin_types, num_samples),
        'Insulin_Dosage': np.random.uniform(2, 40, num_samples)  # Dosage between 2 to 40 units
    }

#    Create a DataFrame
    df = pd.DataFrame(data)

    # Step 2: Build and Train the Model

    # One-hot encoding the Insulin_Type
    encoder = OneHotEncoder(sparse_output=False)
    insulin_encoded = encoder.fit_transform(df[['Insulin_Type']])
    insulin_df = pd.DataFrame(insulin_encoded, columns=encoder.get_feature_names_out(['Insulin_Type']))

    # Combine the features
    X = pd.concat([df[['Blood_Sugar_Level']], insulin_df], axis=1)
    y = df['Insulin_Dosage']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions on test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Step 3: Manual Input and Prediction

    # Get manual input from the user
    blood_sugar_level = float(input("Enter your blood sugar level (mg/dL): "))
    insulin_type = input(f"Enter your insulin type (choose from {insulin_types}): ")

    # Creating a new input DataFrame with the manual inputs
    new_data = {
        'Blood_Sugar_Level': [blood_sugar_level],
        'Insulin_Type': [insulin_type]
    }
    new_input_df = pd.DataFrame(new_data)

    # Transform the new input using the same encoder
    new_input_encoded = encoder.transform(new_input_df[['Insulin_Type']])
    new_input_df_encoded = pd.DataFrame(new_input_encoded, columns=encoder.get_feature_names_out(['Insulin_Type']))

    # Combine the transformed features with the Blood_Sugar_Level
    new_input_final = pd.concat([new_input_df[['Blood_Sugar_Level']], new_input_df_encoded], axis=1)

    # Predicting the dosage
    predicted_dosage = model.predict(new_input_final)
    predicted_dosage_rounded = round(predicted_dosage[0], 2)  # Rounding to 2 decimal places

    print(f"Predicted Insulin Dosage: {predicted_dosage_rounded} units")

    
pickle.dumb()