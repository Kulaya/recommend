import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    st.title("College Recommendation System")

    # Upload dataset
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the dataset
        dataset = pd.read_csv(uploaded_file)

        # Select target column
        target_column = st.sidebar.selectbox("Select Target Column", dataset.columns)

        # Split features and target
        X = dataset.drop(target_column, axis=1)
        y = dataset[target_column]

        # Convert grades to numerical values using label encoding
        label_encoder = LabelEncoder()
        for column in X.columns:
            X[column] = label_encoder.fit_transform(X[column])

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the KNN model
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        st.write("Model Training Complete!")

        # Ask user to input grades
        st.header("Enter Your Grades")
        user_grades = {}
        for column in X.columns:
            grade = st.selectbox(f"What is your grade in {column}?", ['A', 'B', 'C', 'D'])
            user_grades[column] = label_encoder.transform([grade])[0]

        # Predict field of interest
        user_data = pd.DataFrame([user_grades])
        predicted_field = knn.predict(user_data)
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
@st.cache
def load_data():
    dataset = pd.read_csv("course_recommendation_dataset.csv")
    return dataset

# Preprocess the dataset
def preprocess_data(dataset):
    X = dataset.drop('Field of Interest', axis=1)
    y = dataset['Field of Interest']

    # Convert grades to numerical values using label encoding
    label_encoder = LabelEncoder()
    for column in X.columns:
        X[column] = label_encoder.fit_transform(X[column])

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, label_encoder

# Train the KNN model
def train_model(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def main():
    st.title("College Recommendation System")

    # Load the dataset
    dataset = load_data()

    # Preprocess the data
    X_train, y_train, label_encoder = preprocess_data(dataset)

    # Train the model
    knn = train_model(X_train, y_train)

    # Ask the user to enter their grades
    st.write("Please enter your grades for the following subjects (A, B, C, or D):")
    grades = {}
    for column in dataset.columns[:-1]:
        grade = st.selectbox(f"What is your grade in {column}?", ['A', 'B', 'C', 'D'])
        grades[column] = label_encoder.transform([grade])[0]

    # Predict the field of interest
    user_data = pd.DataFrame([grades])
    predicted_field = knn.predict(user_data)
    st.write("\nBased on your grades, the predicted field of interest is:", predicted_field[0])

if __name__ == "__main__":
    main()

        st.success(f"Based on your grades, the predicted field of interest is: {predicted_field[0]}")

if __name__ == "__main__":
    main()
