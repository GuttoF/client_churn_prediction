# Model Deploy using Streamlit

# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Path
path = '/Users/gutto/Repos/client_churn_prediction/'

# Importing Model
model = CatBoostClassifier()
model.load_model(path + 'models/model.cbm')

# Title
st.title('Client Churn Prediction')
st.write('*This is a simple app to predict client churn.*')
st.write('Dataset is avaliable in [Kaggle](https://www.kaggle.com/datasets/mervetorkan/churndataset).')
