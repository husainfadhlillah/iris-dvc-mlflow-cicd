import pandas as pd
import yaml
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib


def run_experiment():
    """
    Melatih, mengevaluasi, dan mencatat eksperimen
    menggunakan MLflow.
    """
    print("Memulai proses run_experiment...")
    # Set nama eksperimen di MLflow
    mlflow.set_experiment("Iris_Classification_RF")

    # Mulai MLflow Run
    with mlflow.start_run() as run:
        # 1. Muat Parameter
        with open("params.yaml", 'r') as f:
            params = yaml.safe_load(f)

        n_estimators = params['train_model']['n_estimators']
        max_depth = params['train_model']['max_depth']
        random_state = params['train_model']['random_state']

        # Log parameter ke MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        print("Parameter berhasil di-log ke MLflow.")

        # 2. Muat Data
        train_df = pd.read_csv("data/processed/train.csv")
        test_df = pd.read_csv("data/processed/test.csv")

        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        # 3. Latih Model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        print("Model berhasil dilatih.")

        # 4. Evaluasi Model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        # Log metrik ke MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        print("Metrik berhasil di-log ke MLflow.")

        # 5. Simpan Metrik ke File (untuk DVC)
        Path("reports").mkdir(parents=True, exist_ok=True)
        metrics = {'accuracy': acc, 'f1_macro': f1}
        with open("reports/metrics.json", 'w') as f:
            json.dump(metrics, f)
        print("Metrik berhasil disimpan di reports/metrics.json")

        # 6. Simpan Model (untuk DVC & MLflow)
        Path("models").mkdir(parents=True, exist_ok=True)
        joblib.dump(model, "models/model.joblib")

        # Log model ke MLflow
        mlflow.sklearn.log_model(model, "model")
        print("Model log ke MLflow & disimpan di models/model.joblib")

        print(f"\nEksperimen selesai. Run ID: {run.info.run_id}")
        print(f"Akurasi: {acc:.4f}, F1-Score: {f1:.4f}")


if __name__ == "__main__":
    run_experiment()