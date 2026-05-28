from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load('placement_model.pkl')

class Student(BaseModel):
    degree_p: float
    etest_p: float
    mba_p: float
    workex: int

@app.get('/')
def home():
    return {'message': 'Placement Prediction API'}

@app.post('/predict')
def predict(student: Student):

    data = pd.DataFrame([[
        student.degree_p,
        student.etest_p,
        student.mba_p,
        student.workex
    ]], columns=[
        'degree_p',
        'etest_p',
        'mba_p',
        'workex'
    ])

    prediction = model.predict(data)

    result = 'Placed' if prediction[0] == 1 else 'Not Placed'

    return {'Prediction': result}