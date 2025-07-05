#knn uncleaned
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and clean dataset
data = pd.read_csv('cleaned_test.csv')
data = data.dropna()

# Convert cholesterol_level to binary (0 = Normal, 1 = High)
data['cholesterol_level'] = (data['cholesterol_level'] >= 200).astype(int)

# Features and target
X = data[['age', 'bmi', 'asthma', 'hypertension']]
y = data['cholesterol_level']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.2f}")
