#linear regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and clean the dataset
data = pd.read_csv('dataset_med.csv')
data = data.dropna()

# Select input and target
X = data[['age']]
y = data['bmi']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model on training data
model = LinearRegression()
model.fit(X_train, y_train)

# Sort the training data for smooth plotting
train_sorted = X_train.copy()
train_sorted['bmi'] = y_train
train_sorted = train_sorted.sort_values(by='age')
X_sorted = train_sorted[['age']]
y_pred_sorted = model.predict(X_sorted)

# --- User Input for Prediction ---
print("Enter Your Age to Predict BMI")
try:
    age = float(input("Enter age: "))
    user_input = pd.DataFrame([[age]], columns=['age'])
    predicted_bmi = model.predict(user_input)[0]
    print(f"\nPredicted BMI: {predicted_bmi:.2f}")
except Exception as e:
    print(f"Invalid input. Please enter a numeric age. Error: {e}")

# --- Plotting Regression Line ---
plt.figure(figsize=(10, 6))
plt.plot(X_sorted['age'], y_pred_sorted, color='blue', linewidth=2, label='Predicted BMI (Train)')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Linear Regression Line: BMI vs Age (Trained on Training Data)')
plt.legend()
plt.grid(True)
plt.show()