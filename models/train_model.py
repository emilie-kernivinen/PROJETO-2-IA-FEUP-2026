import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("../data/synthetic_spotify_users.csv")

# Features and target
X = data.drop("churn", axis=1)
y = data["churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ), 

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Support Vector Machine": SVC(
        probability=True,
        class_weight="balanced",
        random_state=42
    )
}

best_model = None
best_accuracy = 0
best_model_name = ""

# Train and evaluate all models
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}")
    print(f"{'='*50}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# Save best performing model
joblib.dump(best_model, "best_churn_model.pkl")

print(f"\nBest model: {best_model_name}")
print(f"Best accuracy: {best_accuracy}")
print("Saved as best_churn_model.pkl")

# Feature importance only for tree-based models
if hasattr(best_model, "feature_importances_"):
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_ # type: ignore
    }).sort_values(by="importance", ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)
else:
    print(f"\n{best_model_name} does not support feature_importances_.")