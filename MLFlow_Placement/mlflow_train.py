
import pandas as pd
import mlflow
import mlflow.sklearn

from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. MLflow configuration
# --------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("PlacementPrediction")

# --------------------------------------------------
# 2. Load and prepare dataset
# --------------------------------------------------
df = pd.read_csv("Placement_Data_Full_Class.csv")

features = ["degree_p", "etest_p", "mba_p", "workex"]
target = "status"

df = df[features + [target]].copy()

df["workex"] = df["workex"].map({"Yes": 1, "No": 0})
df["status"] = df["status"].map(
    {"Placed": 1, "Not Placed": 0}
)

df = df.dropna()

X = df[features]
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 3. Train model and track experiment
# --------------------------------------------------
with mlflow.start_run(run_name="RidgeClassifier_Drift_Analysis"):

    model = RidgeClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    # Log parameters
    mlflow.log_param("model", "RidgeClassifier")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("dataset_rows", len(df))
    mlflow.log_param("feature_count", len(features))

    # Log evaluation metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    print("\nMODEL EVALUATION")
    print("----------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # --------------------------------------------------
    # 4. Prepare reference and simulated current data
    # --------------------------------------------------
    reference, current = train_test_split(
        X,
        test_size=0.3,
        random_state=24
    )

    current = current.copy()

    # Simulate an intentional shift in numeric features.
    # This is a test scenario, not naturally observed drift.
    shifted_features = ["degree_p", "etest_p", "mba_p"]

    for column in shifted_features:
        current[column] = (current[column] + 8).clip(upper=100)

    # --------------------------------------------------
    # 5. Detect numeric feature drift using KS test
    # --------------------------------------------------
    results = []

    for column in shifted_features:
        statistic, p_value = ks_2samp(
            reference[column],
            current[column]
        )

        drift_detected = p_value < 0.05

        results.append({
            "Feature": column,
            "KS_Statistic": statistic,
            "P_Value": p_value,
            "Drift_Detected": drift_detected,
            "Reference_Mean": reference[column].mean(),
            "Current_Mean": current[column].mean()
        })

        # Log drift metrics in MLflow
        mlflow.log_metric(
            f"{column}_ks_statistic", float(statistic)
        )
        mlflow.log_metric(
            f"{column}_p_value", float(p_value)
        )
        mlflow.log_metric(
            f"{column}_drift_detected",
            int(drift_detected)
        )

    drift_report = pd.DataFrame(results)
    drift_report.to_csv("drift_results.csv", index=False)

    # Log drift report as an MLflow artifact
    mlflow.log_artifact("drift_results.csv")

    # Log the trained model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

    print("\nDATA DRIFT ANALYSIS")
    print("------------------")
    print(drift_report.to_string(index=False))

    if drift_report["Drift_Detected"].any():
        print("\nResult: Drift detected in one or more features.")
    else:
        print("\nResult: No statistically significant drift detected.")

    print("\nDrift report saved to drift_results.csv")
    print("MLflow experiment: PlacementPrediction")
    print("Run completed successfully.")
