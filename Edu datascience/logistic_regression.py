#logistic regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('C:\Users\Stephina Jose\Desktop\Edu datascience\dataset_med.csv')

# Features and target
X = data[['age', 'bmi', 'asthma', 'hypertension']]
y = data['survived']  # Predicting survival

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data with stratify to ensure both survival classes are represented
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict on test set
y_pred = logreg.predict(X_test)

# --- USER INPUT SECTION ---
print("\nEnter the following details to predict survival:")

try:
    age = float(input("Age: "))
    bmi = float(input("BMI: "))
    asthma = int(input("Asthma (0 = No, 1 = Yes): "))
    hypertension = int(input("Hypertension (0 = No, 1 = Yes): "))

    # Prepare and scale input
    user_data = pd.DataFrame([[age, bmi, asthma, hypertension]], columns=['age', 'bmi', 'asthma', 'hypertension'])
    user_scaled = scaler.transform(user_data)

    # Predict
    user_prediction = logreg.predict(user_scaled)
    result = "Survived" if user_prediction[0] == 1 else "Did Not Survive"
    print(f"\nPredicted Survival Outcome: {result}")

except Exception as e:
    print(f"Invalid input. Error: {e}")