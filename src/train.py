"""
src/train.py

Training script for Random Forest Classifier on Iris dataset.
Integrated with MLflow for experiment tracking and hyperparameter logging,
and saves the trained model to models/model.pkl using joblib.
"""

import os
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_params(params_path="params.yaml"):
    """Load hyperparameters from params.yaml file."""
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params["train"]


def main():
    """Main training routine with MLflow logging and model persistence."""
    print("--- Starting Random Forest Classifier Training ---")

    # 1. Load hyperparameters from params.yaml
    params = load_params()
    n_estimators = params["n_estimators"]
    max_depth = params["max_depth"]
    random_state = params["random_state"]
    test_size = params["test_size"]

    print(f"Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}, "
          f"random_state={random_state}, test_size={test_size}")

    # 2. Read dataset
    data_path = "data/iris.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run create_dataset.py first!"
        )

    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 4. Initialize MLflow experiment tracking URI & experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Iris_Random_Forest_Experiment")

    with mlflow.start_run():
        # 5. Train Random Forest Classifier
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        rf.fit(X_train, y_train)

        # 6. Make predictions & calculate metrics
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"\nModel Evaluation Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        # 7. Log hyper-parameters to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)

        # 8. Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # 9. Log model artifact to MLflow
        mlflow.sklearn.log_model(rf, artifact_path="random_forest_model")
        print("Logged parameters, metrics, and model artifact to MLflow.")

        # 10. Save trained model to disk using joblib
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        model_filepath = os.path.join(models_dir, "model.pkl")
        joblib.dump(rf, model_filepath)
        print(f"Trained model saved locally to {model_filepath}")

    print("--- Training completed successfully ---")


if __name__ == "__main__":
    main()
