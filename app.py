# Step 1: Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Step 2: Load dataset (GitHub CSV version)
url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
iris = pd.read_csv(url)

# Step 3: Data exploration
print("First 5 rows:\n", iris.head())
print("\nMissing values:\n", iris.isnull().sum())

# Step 4: Visualization
sns.pairplot(iris, hue="species")
plt.show()

# Step 5: Prepare features & target
X = iris.drop("species", axis=1)
y = iris["species"]

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Compare multiple models using cross-validation
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

print("\nModel Cross-Validation Scores:")
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"{name}: {scores.mean():.4f}")

# Step 9: Hyperparameter tuning for the best model (Random Forest here)
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [2, 4, 6, None]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

print("\nBest Parameters:", grid.best_params_)
best_model = grid.best_estimator_

# Step 10: Evaluate on test set
y_pred = best_model.predict(X_test_scaled)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 11: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix Heatmap")
plt.show()

# Step 12: Save and Load Model
joblib.dump(best_model, "iris_model.pkl")
print("\nModel saved as 'iris_model.pkl'")

loaded_model = joblib.load("iris_model.pkl")
print("\nLoaded Model Prediction Example:", loaded_model.predict([X_test_scaled[0]]))
