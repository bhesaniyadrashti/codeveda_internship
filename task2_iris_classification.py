import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

# Load dataset
df = pd.read_csv("iris.csv")

# Features
X = df.drop("species", axis=1)

# Target
y = df["species"]

# Convert species to numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

print("\n===== CLASSIFICATION RESULTS =====\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(name)

    print("Accuracy:",
          accuracy_score(y_test, y_pred))

    print("Precision:",
          precision_score(
              y_test,
              y_pred,
              average="weighted"
          ))

    print("Recall:",
          recall_score(
              y_test,
              y_pred,
              average="weighted"
          ))

    print(classification_report(y_test, y_pred))

    print("-" * 40)