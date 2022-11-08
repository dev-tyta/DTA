# importing python modules.
import streamlit as st
import joblib
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

# creating input feature for data
name = st.text("Patient's Name: ")
gender = st.selectbox(label="Patient's Gender: ", options=["Male", "Female"])
age = st.slider(label="Patient's Age: ", min_value=0, max_value=100)
chol = st.slider(label="Patient's Cholesterol Level(mg/dL): ", min_value=40, max_value=200)
glucose = st.slider(label="Patient's Sugar Level(mg/dL): ", min_value=40, max_value=250)
height_cm = st.number_input(label="Patient's Height(cm): ")
height = height_cm * 0.393701
weight_kg = st.number_input("Patient's Weight in(kg): ")
weight = weight_kg * 2.205
bmi = st.slider(label="Patient's Body Mass:", min_value=0, max_value=50)
hdl_chol = st.slider(label="Patient's HDL Cholesterol(mg/dL): ", min_value=0, max_value=100)
chol_hdl_ratio = chol / hdl_chol
waist = st.number_input("Patient's Waist Size(inches): ")
hip = st.number_input("Patient's Hip Size(inches): ")
waist_hip_ratio = waist / hip
systolic_bp = st.number_input(label="Patient's Systolic Blood Pressure(mmHg): ")
diastolic_bp = st.number_input(label="Patient's Diastolic Blood Pressure(mmHg): ")
