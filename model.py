import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import joblib

s = pd.read_csv("C:/Users/josep/p2_final_project/Data/social_media_usage.csv")

def clean_sm(x):
    x = np.where(x==1,1,0)
    return x

ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"]<=9,s["income"], np.nan),
    "education":np.where(s["educ2"]<=8, s["educ2"], np.nan),
    "parent":np.where(s["par"]==1, 1, 0),
    "married":np.where(s["par"]==1, 1, 0),
    "female":np.where(s["gender"]==2, 1, 0),
    "age":np.where(s["age"]<=98,s["age"], np.nan)})    

ss = ss.dropna()

ss = ss.astype(int)

Y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, Y_train, Y_test = train_test_split(X,
Y,
stratify=Y,
test_size=.2,
random_state=101)

lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, Y_train)

y_pred = lr.predict(X_test)

lr.score(X_train, Y_train)

accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy}")

joblib.dump(lr, "lr_model.sav")