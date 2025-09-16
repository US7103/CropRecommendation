import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pickle


df=pd.read_csv('Crop_recommendation.csv')

x=df.drop(['label'],axis='columns')


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
mat=df['label']
le.fit(mat)
df['ln_n']=le.transform(mat)
df
y=df['ln_n']


model=RandomForestClassifier()
from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.2)
model.fit(x_train,y_train)

with open('crop_recommend.pkl','wb') as f:
    pickle.dump((model,le),f)