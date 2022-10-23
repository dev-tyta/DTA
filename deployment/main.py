# importing python modules.
import streamlit as st
import pickle
import joblib

# loading pickle files gotten from model
lightgbm_pickle = open("lightgbm", "rb")
lgbm_model = joblib.load(lightgbm_pickle)
