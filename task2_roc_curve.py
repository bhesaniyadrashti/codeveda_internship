import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

df = pd.read_csv("iris.csv")

X = df.drop("species", axis=1)
y = df["species"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Save binarized version only for ROC
y_bin = label_binarize(y, classes=[0, 1, 2])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

y_score = model.predict_proba(X_test)

# Binarize test labels for ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

fpr, tpr, _ = roc_curve(
    y_test_bin.ravel(),
    y_score.ravel()
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()