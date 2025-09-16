import dagshub
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
    classification_report
)

# --- Init DagsHub ---
dagshub.init(repo_owner="jasmeinalbr", repo_name="Membangun_model", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/jasmeinalbr/Membangun_model.mlflow")
mlflow.set_experiment("Loan Prediction - Advanced")

# --- Load data preprocess ---
X_train = pd.read_csv("dataset_preprocessing/train_processed.csv")
X_test = pd.read_csv("dataset_preprocessing/test_processed.csv")
y_train = pd.read_csv("dataset_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("dataset_preprocessing/y_test.csv").values.ravel()

# --- Candidate models ---
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

# --- Hyperparameter grids ---
param_grids = {
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    },
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    }
}

# --- Loop over models ---
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        
        # Grid search
        grid = GridSearchCV(model, param_grids[model_name], cv=5, scoring="accuracy")
        grid.fit(X_train, y_train)

        # Best model & params
        best_model = grid.best_estimator_
        best_params = grid.best_params_

        # Evaluate
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # --- Manual Logging ---
        # log hyperparameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # log metrics
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # --- Artefacts ---
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        cm_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Feature importance (RandomForest only)
        if model_name == "RandomForest":
            importances = best_model.feature_importances_
            plt.figure(figsize=(8,4))
            sns.barplot(x=importances, y=X_train.columns)
            plt.title("Feature Importances")
            plt.tight_layout()
            fi_path = f"feature_importance_{model_name}.png"
            plt.savefig(fi_path)
            plt.close()
            mlflow.log_artifact(fi_path)

        # Classification report (JSON)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = f"classification_report_{model_name}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact(report_path)

        # Simpan model ke file lokal
        model_path = f"{model_name}_best_model.pkl"
        joblib.dump(best_model, model_path)

        # Log ke DagsHub sebagai artifact
        mlflow.log_artifact(model_path)

        print(f"✅ {model_name} | Best params: {best_params} | "
              f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        