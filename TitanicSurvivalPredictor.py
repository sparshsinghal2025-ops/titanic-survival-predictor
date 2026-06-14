from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier  # Changed to Classifier
from sklearn.linear_model import LogisticRegression  # Changed to Classifier
from sklearn.metrics import accuracy_score, f1_score  # Classification metrics
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
from xgboost import XGBClassifier  # Changed to Classifier
import numpy as np
import pandas as pd
import pickle

warnings.filterwarnings("ignore")

# ===== 1. LOAD DATA =====
# Updated path and target logic for Titanic
df = pd.read_csv(r"E:\Beginner ML projects datasets\Titanic-Dataset.csv")


# Clean specific Titanic missing values before ColumnTransformer
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Age"] = df["Age"].fillna(df["Age"].median())

df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

df["Embarked"] = df["Embarked"].fillna("S")

# Clean messy Excel shifts in Sex column
df["Sex"] = df["Sex"].astype(str).str.strip().str.lower()
df["Sex"] = df["Sex"].map({"male": "male", "m": "male", "female": "female", "f": "female"})
df["Sex"] = df["Sex"].fillna("male")

# ===== 2. SPLIT DATA =====
# Removed regression quantile outlier filtering as it can drop valid survival signals

# Drop non-predictive text IDs
X = df.drop(columns=["PassengerId", "Survived", "Name", "Ticket", "Cabin"], errors="ignore")
y = df["Survived"].astype(int)

# Categorical columns matching the Titanic structure
categorical_cols = ["Sex", "Embarked"]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="first"), categorical_cols)],
    remainder="passthrough",
)

# Convert all text columns into numbers
X_processed = preprocessor.fit_transform(X)

# Split the processed numeric data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 3. MODELS + TUNING =====
# Swapped out all Regressors for Classifiers
# Define hyperparameter lists explicitly to avoid rendering glitches
rf_estimators = [100, 200, 300]
rf_splits = [2, 5, 10]
xgb_estimators = [100, 200]
xgb_depths = [3, 5, 7]

models = {
    "Logistic": {
        "model": LogisticRegression(), 
        "params": {}
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": rf_estimators,
            "max_depth": [5, 10, None],
            "min_samples_split": rf_splits
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss"),
        "params": {
            "n_estimators": xgb_estimators,
            "max_depth": xgb_depths,
            "learning_rate": [0.01, 0.1, 0.3]
        }
    }
}

results = {}

print("Training classification models... grab chai ☕")
print("-" * 40)

for name, config in models.items():
    model = config["model"]
    params = config["params"]

    if params:
        # Changed scoring metric to 'accuracy' for classification optimization
        search = RandomizedSearchCV(
            model, params, n_iter=10, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
        )
        search.fit(X_train_scaled, y_train)
        best_model = search.best_estimator_
        print(f"{name}: Tuning done. Best params = {search.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train_scaled, y_train)
        print(f"{name}: Trained")

    # Evaluate using Classification metrics
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {"Accuracy": acc, "F1_Score": f1, "Model": best_model}

# ===== 4. SHOW WINNER =====
print("\n" + "=" * 40)
print("FINAL RESULTS")
print("=" * 40)

for name, scores in sorted(results.items(), key=lambda x: x[1]["Accuracy"], reverse=True):
    print(f"{name:15} | Accuracy = {scores['Accuracy']:.4f} | F1-Score = {scores['F1_Score']:.4f}")

winner_name = max(results, key=lambda x: results[x]["Accuracy"])
print("\n🏆 WINNER:", winner_name)
print(f"Best Accuracy: {results[winner_name]['Accuracy']:.4f}")
print(f"Use this model: results['{winner_name}']['Model']")

# Save artifacts tailored for Titanic
artifacts = {
    "model": results[winner_name]["Model"],
    "preprocessor": preprocessor,
    "scaler": scaler,
}

with open("TitanicSurvival.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\nModel, preprocessor, and scaler saved cleanly to TitanicSurvival.pkl")
