import streamlit as st
import pickle
import requests
import pandas as pd
import os

path = os.path.dirname(__file__)

data=pickle.load(open('data.pkl', 'rb'))
model=pickle.load(open('model.pkl', 'rb'))

st.title("Air Quality Forecasting")



df1=pd.DataFrame()
df1={'Gujarat':0, 'Mizoram':1, 'Andhra Pradesh':2, 'Punjab':3, 'Karnataka':4,
       'Madhya Pradesh':5, 'Odisha':6, 'Chandigarh':7, 'Tamilnadu':8, 'Delhi':9,
       'Kerala':10, 'Haryana':11, 'Assam':12, 'Telengana':13, 'Rajasthan':14,
       'Jharkhand':15, 'kerala':16, 'West Bengal':17, 'Uttar Pradesh':18,
       'Maharashtra':19, 'Bihar':20, 'Meghalaya':21}

name1=['Gujarat', 'Mizoram', 'Andhra Pradesh', 'Punjab', 'Karnataka',
       'Madhya Pradesh', 'Odisha', 'Chandigarh', 'Tamilnadu', 'Delhi',
       'Kerala', 'Haryana', 'Assam', 'Telengana', 'Rajasthan',
       'Jharkhand', 'kerala', 'West Bengal', 'Uttar Pradesh',
       'Maharashtra', 'Bihar', 'Meghalaya']

df2=pd.DataFrame()
df2={'Ahmedabad':0, 'Aizawl':1, 'Amaravati':2, 'Amritsar':3, 'Bengaluru':4,
       'Bhopal':5, 'Brajrajnagar':6, 'Chandigarh':7, 'Chennai':8, 'Coimbatore':9,
       'Delhi':10, 'Ernakulam':11, 'Gurugram':12, 'Guwahati':13, 'Hyderabad':14,
       'Jaipur':15, 'Jorapokhar':16, 'Kochi':17, 'Kolkata':18, 'Lucknow':19, 'Mumbai':20,
       'Patna':21, 'Shillong':22, 'Talcher':23, 'Thiruvananthapuram':24,
       'Visakhapatnam':25}

name2=['Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru',
       'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore',
       'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad',
       'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow', 'Mumbai',
       'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram',
       'Visakhapatnam']


df3=pd.DataFrame()
df3={'Jannuary':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
name3=['Jannuary','February','March','April','May','June','July','August','September','October','November','December']

option1 = st.selectbox(
     'Select a State of your choice',
     (name1))

st.write('You selected:', df1[option1])

option2 = st.selectbox(
     'Select a City of your choice',
     (name2))

st.write('You selected:', df2[option2])

option3 = st.selectbox(
     'Year',
     ([2021, 2022,2023,2024,2025]))

st.write('You selected:', option3)


option4 = st.selectbox(
     'Month',
     (name3))

st.write('You selected:', df3[option4])


def forecasting():
     from lightgbm import LGBMRegressor
     
     x_predict = [df1[option1],df2[option2],option3,df3[option4]]
     import numpy as np
     x_predict = np.array(x_predict).reshape(1,-1)
     predicted_value = model.predict(x_predict)    
     return predicted_value


if st.button('Forecast'):
     pred= forecasting()
     st.write('Forecasted air quality index is',pred)






