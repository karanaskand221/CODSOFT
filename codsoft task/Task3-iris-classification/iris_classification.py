import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  # ← BUG FIXED: Missing import
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load CSV
df = pd.read_csv("IRIS.csv")

# Quick checks
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df["species"].value_counts())

# Features (X) and target (y)
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", acc)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Prediction function
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    data_scaled = scaler.transform(data)
    return model.predict(data_scaled)[0]

# Example prediction
print("\nPrediction test:")
print(predict_species(5.1, 3.5, 1.4, 0.2))  # Should output: Iris-setosa

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, y_pred_knn)
print(f"\nKNN Accuracy: {knn_acc:.4f}")
