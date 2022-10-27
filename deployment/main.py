# importing python modules.
import streamlit as st
import pickle as pkl

# loading pickle files gotten from model
lightgbm_pickle = open("./lightgbm.pkl", "rb")
lgbm_model = pkl.load(lightgbm_pickle)

score = lgbm_model.score()

# create a function to
st.title("Diabetes Prediction App")
st.write("Test 1")
st.write(score)
