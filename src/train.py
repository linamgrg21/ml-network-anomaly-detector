import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")

df = pd.read_csv("data/KDDTrain+.txt", header=None)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("Preprocessing data...")

# NSL-KDD format:
# Last column = difficulty score
# Second last column = attack label
X = df.iloc[:, :-2]
y = df.iloc[:, -2]

# Binary label: normal = 0, attack = 1
y = y.apply(lambda label: 0 if label == "normal" else 1)

# Convert categorical columns like tcp, udp, http, SF into numeric columns
X = pd.get_dummies(X)

# Fix sklearn column name issue
X.columns = X.columns.astype(str)

print("Feature shape after encoding:", X.shape)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Making predictions...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n=== RESULTS ===")
print("Accuracy:", accuracy)
print(report)

os.makedirs("results", exist_ok=True)

with open("results/classification_report.txt", "w") as file:
    file.write("Network Anomaly Detection Results\n")
    file.write("=================================\n\n")
    file.write(f"Accuracy: {accuracy}\n\n")
    file.write(report)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"]
)

plt.title("Network Anomaly Detection Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

joblib.dump(model, "results/model.pkl")

print("Classification report saved to results/classification_report.txt")
print("Confusion matrix saved to results/confusion_matrix.png")
print("Model saved to results/model.pkl")
print("Project completed successfully!")