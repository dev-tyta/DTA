# importing python modules.
import streamlit as st
import joblib
import pickle
import numpy as np

# loading pickle files gotten from model
lightgbm_pickle = open(r"C:\Users\Testys\Documents\GitHub\Streamlit-Deployment corrupted\deployment\lightgbm.pickle",
                       "rb")
lgbm_model = joblib.load(lightgbm_pickle)

var = [200, 80, 67, 2.99, 20, 0.0, 60, 160, 24.6, 120, 60, 32, 38, 0.84, 2.67]
var_names = ['cholesterol', 'glucose', 'hdl_chol', 'chol_hdl_ratio', 'age'
            , 'gender', 'weight', 'height', 'bmi', 'systolic_bp', 'diastolic_bp', 'waist', 'hip'
            , 'waist_hip_ratio', 'diabetes', 'height_weight']


def predict(var_name):
    pred = [var_name]
    np_pred = np.array(pred)
    score = lgbm_model.predict(np_pred)
    return score


# test case
result = predict(var)
print(result)

# create a function to
st.title("Diabetes Prediction App")
st.write("Test 1")
st.write(result)

# creating input feature for data
name = st.text("Name: ")
gender = st.selectbox(label="Gender: ", options=["Male", "Female"])

