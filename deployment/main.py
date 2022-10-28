# importing python modules.
import streamlit as st
import pickle as pkl
import numpy as np

# loading pickle files gotten from model
lightgbm_pickle = open("./lightgbm.pickle", "rb")
lgbm_model = pkl.load(lightgbm_pickle)


def predict(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15):
    pred = [var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15]
    np_pred = np.array(pred)
    score = lgbm_model.predict(np_pred)
    return score


# create a function to
st.title("Diabetes Prediction App")
st.write("Test 1")
st.write(score)
