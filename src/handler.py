import os
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List
import pandas as pd
from catboost import CatBoostClassifier
from churn_prediction import TopBankChurnPrediction

path = '/Users/guttofranca/Repos/client_churn_prediction/src/model'
#path = 'model/'

# loading model
model = pickle.load(open(path + 'cross_sell.pkl', 'rb'))
model = CatBoostClassifier()
model = model.load_model(path + 'models/model.cbm')
print('Model loaded')

print('\n----- INITIALIZING SERVER -----\n')

# FastAPI
app = FastAPI()

class Item(BaseModel):
    # Define the structure of the input data using Pydantic BaseModel
    # This will automatically validate the input JSON
    # Adjust the fields as per your data structure
    field1: float
    field2: float
    # Add more fields as needed

@app.post("/predict")
async def healthinsurance_prediction(item: Item):
    # Convert input Pydantic model to a Pandas DataFrame
    df_raw = pd.DataFrame(item.dict(), index=[0])

    # TopBankChurnPrediction class
    pipeline = TopBankChurnPrediction()

    # loading data
    df1 = pipeline.loading_data(df_raw)

    # build_features
    df2 = pipeline.build_features(df1)

    # preprocess_data
    df3 = pipeline.preprocess_data(df2)

    # prediction
    df_response = pipeline.get_prediction(model, df3)

    # Convert the Pandas DataFrame response to JSON
    response_json = jsonable_encoder(df_response)

    return JSONResponse(content=response_json)

@app.post("/predict_batch")
async def healthinsurance_prediction_batch(items: List[Item]):
    # Convert input Pydantic model to a Pandas DataFrame
    df_raw = pd.DataFrame(items.dict(), index=[0])

    # Healthinsurance class
    pipeline = TopBankChurnPrediction()

    # loading data
    df1 = pipeline.loading_data(df_raw)

    # build_features
    df2 = pipeline.build_features(df1)

    # preprocess_data
    df3 = pipeline.preprocess_data(df2)

    # prediction
    df_response = pipeline.get_prediction(model, df3)

    # Convert the Pandas DataFrame response to JSON
    response_json = jsonable_encoder(df_response)

    return JSONResponse(content=response_json)

if __name__ == '__main__':
    # Run using uvicorn instead of app.run
    uvicorn.run(app, host="0.0.0.0", port=8000)
