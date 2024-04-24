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

        st.success(f"Based on your grades, the predicted field of interest is: {predicted_field[0]}")

if __name__ == "__main__":
    main()
