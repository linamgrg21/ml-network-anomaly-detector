import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("Loading dataset...")

data = pd.read_csv("data/KDDTrain+.txt", header=None)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Preprocessing data...")

y = y.apply(lambda x: 0 if x == "normal" else 1)

X = pd.get_dummies(X)
X.columns = X.columns.astype(str)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Making predictions...")

y_pred = model.predict(X_test)

print("\n=== RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))