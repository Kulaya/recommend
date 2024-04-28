import pandas as pd
import streamlit as st
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

# Streamlit application
st.title("Course Recommendation System")

st.write("Welcome to the Course Recommendation System!")
st.write("Please enter your grades for the following subjects (A, B, C, D, or F):")

# Dictionary to map field of interest to recommended course
recommended_courses = {
    "Domestic and Industrial Electricity": "Electrical and Industrial Automation Engineering",
    "Computer": "Electrical and Computer Engineering",
    "Electronics": "Electronics and Telecommunication Engineering",
    "Networks": "Electrical and Computer Engineering",
    "Solar Electricity": "Electrical and Renewable Energy"
}

grades = {}
for idx, column in enumerate(X.columns):
    while True:
        grade = st.text_input(f"What is your grade in {column}? ", key=f"grade_{idx}")
        if grade.upper() in ["A", "B", "C", "D", "F"]:
            if grade.upper() != "F":
                grades[column] = label_encoder.transform([grade.upper()])[0]
            break
        else:
            st.write("Invalid grade. Please enter A, B, C, D, or F.")

if grades:
    # Predict the field of interest
    user_data = pd.DataFrame([grades])
    predicted_field = knn.predict(user_data)

    # Map predicted field to recommended course
    recommended_course = recommended_courses.get(predicted_field[0], "Unknown")

    if recommended_course != "Unknown":
        st.write("\nBased on your grades, the recommended course is:", recommended_course)
    else:
        st.write("\nBased on your grades, no recommended course is available.")
