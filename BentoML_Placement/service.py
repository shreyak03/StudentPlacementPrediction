from __future__ import annotations

import bentoml
import pandas as pd
import joblib

# Load model
model = joblib.load("placement_model.pkl")

# Create service
@bentoml.service
class PlacementService:

    @bentoml.api
    def predict(
        self,
        degree_p: float,
        etest_p: float,
        mba_p: float,
        workex: int
    ) -> dict:

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

        result = "Placed" if prediction[0] == 1 else "Not Placed"

        return {"Prediction": result}