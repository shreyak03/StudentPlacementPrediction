import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow

# Load dataset
df = pd.read_csv('Placement_Data_Full_Class.csv')

# Select required columns
df = df[[
    'degree_p',
    'etest_p',
    'mba_p',
    'workex',
    'status'
]]

# Convert categorical columns
df['workex'] = df['workex'].map({
    'Yes': 1,
    'No': 0
})

df['status'] = df['status'].map({
    'Placed': 1,
    'Not Placed': 0
})

# Features and target
X = df[['degree_p', 'etest_p', 'mba_p', 'workex']]
y = df['status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Start MLflow run
with mlflow.start_run():

    # Train model
    model = LogisticRegression()

    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("model", "LogisticRegression")

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    print("Accuracy:", accuracy)