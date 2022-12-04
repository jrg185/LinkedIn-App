import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st


def predict(data):
    mod = joblib.load("lr_model.sav")
    return mod.predict(data)

def probs(data):
    prob = joblib.load("lr_model.sav")
    return prob.predict_proba(data)


st.markdown("# LinkedIn User Prediction App")
st.markdown("## This app is to predict whether or not you are a LinkedIn User based on multiple variables")

st.header("User Inputs")
col3, col4 = st.columns(2)
with col3:
    st.markdown("***Income Reference (in USD)***")
    st.markdown(" 1 < 10k")
    st.markdown(" 2 10k - 20k")
    st.markdown(" 3 20k - 30k")
    st.markdown(" 4 30k - 40k")
    st.markdown(" 5 40k - 50k")
    st.markdown(" 6 50k - 75k")
    st.markdown(" 7 75k - 100k")
    st.markdown(" 8 100k - 150k")
    st.markdown(" 9 150k or more")

with col4: 
    st.markdown("***Education Level Reference***")
    st.markdown(" 1 Less than Highschool (Grades 1-8 or no formal schooling)")
    st.markdown(" 2 High School Incomplete (Grades 9-11 or Grade 12 with NO diploma)")
    st.markdown(" 3 High School Graduate (Grade 12 with diploma or GED certificate)")
    st.markdown(" 4 Some college, no degree (includes some community college)")
    st.markdown(" 5 Two-year associate degree from a college or university")
    st.markdown(" 6 Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)")
    st.markdown(" 7 ome postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)")
    st.markdown(" 8 Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)")
    


col1, col2 = st.columns(2)
with col1:
    income = st.slider('Income Level(1-9)',1,9,1)
    education = st.slider('Education Level(1-8)', 1,8,1)
    parent = st.slider('Parent(1 = yes, 0 = no)',0,1,1)

with col2:
    married = st.slider('Married (1 = yes,0 = No)',0,1,1)
    female = st.slider('Gender(0 = Male, 1 = Female)',0,1,1)
    age = st.slider('Age',1,98,1)



if st.button("Predict if LinkedIn User"):
    result = predict(np.array([[income, education, parent, married, female, age]]))
    prob = probs(np.array([[income, education, parent, married, female, age]]))
    st.text(f"Prediction: **{result[0]}** (1 is 'User', 0 is 'NOT a User')")
    st.text(f"Probability that this person is a LinkedIn user: {prob[0][1]}")



