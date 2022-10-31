# importing python modules.
import streamlit as st
import pickle as pkl
import numpy as np

# loading pickle files gotten from model
lightgbm_pickle = open("./lightgbm.pickle", "rb")
lgbm_model = pkl.load(lightgbm_pickle)

var = [200, 80, 67, 2.99, 20, 1.0, 60, 160, 24.6, 120, 60, 32, 38, 0.84]


def predict(var_name):
    pred = [var_name]
    np_pred = np.array(pred)
    score = lgbm_model.predict(np_pred)
    return score


result = predict(var)
print(result)

# create a function to
st.title("Diabetes Prediction App")
st.write("Test 1")
st.write(result)
