import pandas as pd
from pycaret.classification import *
import joblib

# Load dataset

df = pd.read_csv('Placement_Data_Full_Class.csv')

# Select important columns

df = df[[
    'degree_p',
    'etest_p',
    'mba_p',
    'workex',
    'status'
]]

# Convert categorical values

df['workex'] = df['workex'].map({
    'Yes': 1,
    'No': 0
})

# Convert target column

df['status'] = df['status'].map({
    'Placed': 1,
    'Not Placed': 0
})

# Setup PyCaret

setup(
    data=df,
    target='status',
    session_id=123,
    verbose=False
)

# Compare models

best_model = compare_models()

# Finalize best model

final_model = finalize_model(best_model)

# Save model

joblib.dump(final_model, 'placement_model.pkl')

print('Model Saved Successfully')