import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading train and test datasets...")

train_df = pd.read_csv("data/KDDTrain+.txt", header=None)
test_df = pd.read_csv("data/KDDTest+.txt", header=None)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

X_train = train_df.iloc[:, :-2]
y_train = train_df.iloc[:, -2]

X_test = test_df.iloc[:, :-2]
y_test = test_df.iloc[:, -2]

y_train = y_train.apply(lambda label: 0 if label == "normal" else 1)
y_test = y_test.apply(lambda label: 0 if label == "normal" else 1)

print("Encoding categorical features...")

combined = pd.concat([X_train, X_test], axis=0)
combined = pd.get_dummies(combined)
combined.columns = combined.columns.astype(str)

X_train_encoded = combined.iloc[:len(X_train)]
X_test_encoded = combined.iloc[len(X_train):]

print("Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_encoded, y_train)

print("Making predictions...")

y_pred = model.predict(X_test_encoded)

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

plt.title("Confusion Matrix - NSL-KDD Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

joblib.dump(model, "results/model.pkl")

print("Classification report saved.")
print("Confusion matrix saved.")
print("Model saved.")
print("Project completed successfully.")