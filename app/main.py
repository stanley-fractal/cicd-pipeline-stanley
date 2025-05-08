
from fastapi import FastAPI
# from app import model
# from app import ShipmentData, Prediction

import joblib
import pandas as pd

model = joblib.load("app/ecommerce-model.pkl")

from pydantic import BaseModel

class ShipmentData(BaseModel):
    Customer_care_calls: int
    Customer_rating: int
    Cost_of_the_Product: float
    Prior_purchases: int
    Discount_offered: float
    Weight_in_gms: float
    # Add one-hot fields like Mode_of_Shipment_Ship: int, etc.

class Prediction(BaseModel):
    prediction: int


def predict_delay(data):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return int(prediction)
app = FastAPI()

@app.post("/predict", response_model=Prediction)
def predict(data: ShipmentData):
    prediction = predict_delay(data)
    return {"prediction": prediction}
