from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('placement_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    degree_p = float(request.form['degree_p'])
    etest_p = float(request.form['etest_p'])
    mba_p = float(request.form['mba_p'])
    workex = int(request.form['workex'])

    data = pd.DataFrame([[
        degree_p,
        etest_p,
        mba_p,
        workex
    ]], columns=[
        'degree_p',
        'etest_p',
        'mba_p',
        'workex'
    ])

    prediction = model.predict(data)

    result = 'Placed' if prediction[0] == 1 else 'Not Placed'

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)