import pandas as pd

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
