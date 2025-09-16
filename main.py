import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

model,le=pickle.load(open('crop_recommend.pkl','rb'))


st.title('Crop recommendation using Random Forest Classifier')

st.write("Enter the soil and weather conditions below to get the best crop suggestion.")


N=st.number_input('Nitrogen',min_value=0, max_value=140, step=1)
P=st.number_input('Phosphorous',min_value=5, max_value=145, step=1)	
K=st.number_input('Potassium',min_value=5, max_value=205, step=1)
temparature=st.number_input('Temparature (C)',min_value=0.0, max_value=50.0, step=0.1)
humidity=st.number_input('Humidity (%)',min_value=0.0, max_value=100.0, step=0.1)
ph=st.number_input('Ph Value',min_value=0.0, max_value=14.0, step=0.1)
rainfall=st.number_input('Rainfall (mm)',min_value=0.0, max_value=900.0, step=0.1)


if st.button('Recommend Crop'):
    features=np.array([[N,P,K,temparature,humidity,ph,rainfall]])
    num=model.predict(features)
    label_r=le.inverse_transform(num)
    st.success(f"Predicted Crop is {label_r[0]}")
