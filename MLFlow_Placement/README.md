
# Student Placement Prediction with MLflow and Data Drift Analysis

## 1. Project Overview

This project implements a Student Placement Prediction model using
machine learning. MLflow is used to track model experiments, parameters,
evaluation metrics, and model artifacts. Statistical data drift analysis
is performed to compare reference data with simulated current data.

## 2. Objectives

- Train a Ridge Classifier for student placement prediction.
- Track machine learning experiments using MLflow.
- Record accuracy, precision, recall, and F1-score.
- Log the trained model as an MLflow artifact.
- Detect numerical feature distribution changes using the
  Kolmogorov-Smirnov (KS) statistical test.
- Generate a CSV report containing drift analysis results.

## 3. Technologies Used

- Python
- Pandas
- Scikit-learn
- MLflow
- SciPy
- SQLite

## 4. Dataset

Dataset file: `Placement_Data_Full_Class.csv`

Features used:
- `degree_p`: Degree percentage
- `etest_p`: Employability test percentage
- `mba_p`: MBA percentage
- `workex`: Work experience

Target variable:
- `status`: Placement status

The categorical columns are converted into numerical values before
training the classifier.

## 5. Project Structure

```text
MLFlow_Placement/
├── mlflow_train.py
├── Placement_Data_Full_Class.csv
├── mlflow.db
├── drift_results.csv
├── requirements.txt
└── README.md
```

## 6. Installation

Install Python 3.11 or a compatible Python version.

Open a terminal in the project directory and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## 7. Execute the Experiment

Run the model training and drift analysis script:

```bash
python mlflow_train.py
```

The script trains a Ridge Classifier, evaluates its predictions,
logs the experiment to MLflow, and generates `drift_results.csv`.

## 8. MLflow Experiment Tracking

The tracking database is configured using:

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("PlacementPrediction")
```

The experiment logs:
- Model name and training parameters
- Accuracy
- Precision
- Recall
- F1-score
- KS statistics and p-values
- Feature-level drift decisions
- Trained model artifact
- Drift analysis CSV artifact

## 9. Launch the MLflow UI

In a separate terminal, run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the following address in a browser:

http://127.0.0.1:5000

Select the `PlacementPrediction` experiment to inspect the recorded runs,
metrics, parameters, and artifacts.

## 10. Data Drift Detection

The experiment uses the two-sample Kolmogorov-Smirnov test to compare
the distributions of the following numerical features:

- `degree_p`
- `etest_p`
- `mba_p`

Significance threshold: 0.05

Interpretation:
- If p-value < 0.05, a statistically significant distribution difference
  is detected.
- If p-value >= 0.05, there is insufficient statistical evidence of
  a distribution difference at the chosen threshold.

The current dataset is intentionally modified by increasing the three
numerical features by eight percentage points, capped at 100. Therefore,
the experiment demonstrates simulated data drift rather than naturally
observed drift.

Results are saved in `drift_results.csv` and logged as an MLflow artifact.

## 11. Experimental Results

Model evaluation results from the recorded run:

| Metric | Value |
|---|---:|
| Accuracy | 69.77% |
| Precision | 77.42% |
| Recall | 80.00% |
| F1-score | 78.69% |

The experiment detected statistically significant drift in all three
tested numerical features in the simulated scenario.

## 12. Conclusion

The project demonstrates machine learning experiment tracking using
MLflow and feature-level data drift detection using the KS test.
Evaluation metrics and drift statistics are recorded for inspection
through the MLflow UI. The drift report provides evidence of distribution
changes in the simulated current dataset.

## 13. Limitations

- The drift scenario is simulated.
- Only three numerical features are tested for drift.
- Detected feature drift does not automatically establish a decline
  in model performance.
- Further evaluation on genuinely new student data is required to
  assess real-world drift and model performance.

## 14. Reproducibility

To reproduce the experiment:

1. Obtain the project files and dataset.
2. Install dependencies using `requirements.txt`.
3. Run `python mlflow_train.py`.
4. Launch the MLflow UI.
5. Inspect the experiment run and drift report.

The SQLite database and generated artifacts can be retained as
experimental evidence. Do not commit sensitive data or unnecessary
local environment files to a public repository.
