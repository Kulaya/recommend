import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
dataset = pd.read_csv("course_recommendation_dataset.csv")

# Split features (grades) and target (field of interest)
X = dataset.drop('Field of Interest', axis=1)
y = dataset['Field of Interest']

# Convert grades to numerical values using label encoding
label_encoder = LabelEncoder()
for column in X.columns:
    X[column] = label_encoder.fit_transform(X[column])

# Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model
knn.fit(X, y)

# Define function to get user grades
def get_user_grades():
    grades = {}
    for column in X.columns:
        while True:
            grade = st.text_input(f"What is your grade in {column}? ")
            if grade.upper() in ["A", "B", "C", "D"]:
                grades[column] = label_encoder.transform([grade.upper()])[0]
                break
            elif grade.upper() == "F":
                return None  # Return None if F is encountered
            else:
                st.warning("Invalid grade. Please enter A, B, C, D, or F.")
    return grades

# Main function for Streamlit app
def main():
    st.title("Course Recommendation System")

    # Get user grades
    st.header("Enter Your Grades")
    grades = get_user_grades()

    # Predict and recommend course
    if grades is not None:
        user_data = pd.DataFrame([grades])
        predicted_field = knn.predict(user_data)[0]

        # Map predicted field to recommended course
        recommended_courses = {
            "Domestic and Industrial Electricity": "Electrical and Industrial Automation Engineering",
            "Computer": "Electrical and Computer Engineering",
            "Electronics": "Electronics and Telecommunication Engineering",
            "Networks": "Electrical and Computer Engineering",
            "Solar Electricity": "Electrical and Renewable Energy"
        }

        recommended_course = recommended_courses.get(predicted_field, "Unknown")
        st.success(f"Based on your grades, the recommended course is: {recommended_course}")
    else:
        st.warning("You have 4 or more F's in your grades. Unable to provide a recommendation.")

if __name__ == "__main__":
    main()
