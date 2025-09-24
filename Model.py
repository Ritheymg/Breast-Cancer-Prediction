import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load dataset from CSV
df = pd.read_csv("data.csv")

# Drop 'id' column if exists
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Map diagnosis to 0 (Benign) and 1 (Malignant) correctly
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

# Drop rows with missing target values if any
df = df.dropna(subset=["diagnosis"])

# Separate features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Impute missing feature values with column mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

# Create 'model' folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model, scaler, and imputer for consistent preprocessing later
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(imputer, open("model/imputer.pkl", "wb"))

print("✅ Model, scaler, and imputer saved in 'model/' folder")

from sklearn.metrics import accuracy_score, classification_report

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# More detailed metrics
print(classification_report(y_test, y_pred))
