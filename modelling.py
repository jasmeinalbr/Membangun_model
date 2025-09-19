import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

# 1. Load dataset hasil preprocessing
X_train = pd.read_csv("dataset_preprocessing/train_processed.csv")
X_test = pd.read_csv("dataset_preprocessing/test_processed.csv")
y_train = pd.read_csv("dataset_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("dataset_preprocessing/y_test.csv").values.ravel()

# 2. Set MLflow Tracking URI (lokal dulu)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Loan Prediction - Basic")

# autolog
mlflow.sklearn.autolog()

# 3. Jalankan experiment MLflow
with mlflow.start_run(run_name="LogisticRegression_Basic"):

    # Model sederhana Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"✅ Test Accuracy: {acc:.4f}")
