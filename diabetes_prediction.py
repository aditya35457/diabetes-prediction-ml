import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("diabetes.csv")

# Display first 5 rows
print("First 5 Rows of Dataset:\n")
print(df.head())

# Features (input data)
X = df.drop("Outcome", axis=1)

# Target (output)
y = df["Outcome"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Example Prediction
# -----------------------------

# Sample patient data
# Format:
# Pregnancies, Glucose, BloodPressure, SkinThickness,
# Insulin, BMI, DiabetesPedigreeFunction, Age

sample_data = [[2, 120, 70, 20, 79, 25.0, 0.5, 33]]

# Predict
prediction = model.predict(sample_data)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("The person is Diabetic")
else:
    print("The person is Not Diabetic")