# Importing libraries
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

import streamlit as st

# Loading the model
model = joblib.load("diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Prediction Model",layout="centered", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Diabetes Prediction model")

# input features
st.sidebar.header("Input features", divider=True)

age = st.sidebar.number_input("Age", value=52, min_value=20, max_value=80, step=1)

education = st.sidebar.number_input("Education", value=1, min_value=1, max_value=4, step=1)

currentSmoker = st.sidebar.number_input("Current Smoker", value=0, min_value=0, max_value=1, step=1)

cigsPerDay = st.sidebar.number_input("Cigaretes Per day", value=0.0, min_value=0.0, max_value=70.0, step=1.0)

BPMeds = st.sidebar.number_input("BPMeds", value=0, min_value=0, max_value=1, step=1)

prevalentHyp = st.sidebar.number_input("Prevalent Hypertension",value=0, min_value=0, max_value=1, step=1)

totChol = st.sidebar.number_input("Total Cholestral", value=178.0, min_value=113.0, max_value=600.0, step=1.0)

sysBP = st.sidebar.number_input("Systol BP", value=160.0, min_value=83.5, max_value=295.0, step=1.0)

diaBP = st.sidebar.number_input("Diastol BP", value=98.0, min_value=48.0, max_value=142.5, step=1.0)

BMI = st.sidebar.number_input("BMI", value=40.11, min_value=15.0, max_value=60.0, step=1.0)

heartRate = st.sidebar.number_input("Heart Rate", value=100.0, min_value=40.0, max_value=150.0, step=1.0)

glucose = st.sidebar.number_input("Glucose level", value=225.0, min_value=40.0, max_value=400.0, step=1.0)

TenYearCHD = st.sidebar.number_input("TenYearCHD", value=0, min_value=0, max_value=1, step=1)

# App data
app_data_array = np.array([[age, education, currentSmoker, 
                            cigsPerDay, BPMeds, 
                            prevalentHyp, totChol, 
                            sysBP, diaBP, BMI, heartRate, glucose, TenYearCHD]])


# Making prediction
if st.sidebar.button("Predict"):
  prediction = model.predict(app_data_array)
  st.write(f"The prediction is: {prediction[0]}")

  if prediction == 1:
    st.write(f"The patient is: {"DIABETIC"}")
  else:
    st.write(f"The patient is: {"NON DIABETIC"}")
    