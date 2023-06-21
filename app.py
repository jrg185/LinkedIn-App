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


st.image(image = "https://upload.wikimedia.org/wikipedia/commons/0/01/LinkedIn_Logo.svg")

st.markdown("# LinkedIn User Prediction App")
st.markdown("### The purpose of this app is to test a Machine Learning classification model, predicting whether or not you are a LinkedIn user based on 6 variables.  All responses are completely anonymous.")

st.header("User Inputs")


col1, col2 = st.columns(2)
with col1:
    income = st.selectbox('What Is Your Income Range?',['<$10k', '$10k-$20k', '$20k-$30k', '$30k-$40k', '$40k-$50k','$50k-$75k','$75k-$100k','$100k-$150k','$150k +'])
    education = st.selectbox('What Is Your Education Level?',['Less Than Highschool','High School - No Diploma','High School Graduate','Some College','Two-Year Degree',
    'Four-Year Degree','Some Postgraduate','Postgraduate Degree'])
    parent = st.selectbox('Are You a Parent?', ['Yes','No'])
if income == '<$10k':
    income = 1
elif income == '$10k-$20k':
    income = 2
elif income == '$20k-$30k':
    income = 3
elif income == '$30k-$40k':
    income = 4
elif income == '$40k-$50k':
    income = 5
elif income == '$50k-$75k':
    income = 6
elif income == '$75k-$100k':
    income = 7
elif income == '$100k-$150k':
    income = 8
else:
    income = 9

if education == 'Less Than Highschool':
    education = 1
elif education =='High School - No Diploma':
    education = 2
elif education == 'High School Graduate':
    education = 3
elif education == 'Some College':
    education = 4
elif education == 'Two-Year Degree':
    education = 5
elif education == 'Four-Year Degree':
    education = 6
elif education == 'Some Postgraduate':
    education = 7
else:
    education = 8

if parent == 'Yes':
    parent = 1
else:
    parent = 0

with col2:
    married = st.selectbox('Are You Married?',['Married','Not Married'])
    female = st.selectbox('Do You Identify as Male or Female?',['Male','Female'])
    age = st.number_input('Input Your Age', min_value=1, max_value=98,step=1)

if married == 'Married':
    married = 1
else:
    married = 0

if female == 'Female':
    female = 1
else:
    female = 0


    
if st.button("Predict If LinkedIn User"):
    result = predict(np.array([[income, education, parent, married, female, age]]))
    prob = probs(np.array([[income, education, parent, married, female, age]]))
    if result==1:
        print(st.text("Prediction: IS a LinkedIn User"))
        st.text(f"Probability that this person IS a LinkedIn user: {round((prob[0][1])*100),2}")
    else:
        print(st.text("Prediction: Is NOT a LinkedIn User"))
        st.text(f"Probability that this person is NOT a LinkedIn user: {round((1-prob[0][1])*100),2}")
    




    
    


 













