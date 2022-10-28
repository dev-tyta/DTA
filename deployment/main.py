# importing python modules.
import streamlit as st
import pickle as pkl
import numpy as np

# loading pickle files gotten from model
lightgbm_pickle = open("./lightgbm.pickle", "rb")
lgbm_model = pkl.load(lightgbm_pickle)


def predict():
    pred = []
    np_pred = np.array(pred)
    score = lgbm_model.predict(np_pred)


# create a function to
st.title("Diabetes Prediction App")
st.write("Test 1")
st.write(score)
