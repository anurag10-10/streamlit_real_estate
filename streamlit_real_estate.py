import streamlit as st
import pickle
import numpy as np
import json

# Load the model
model = pickle.load(open('banglore_home_prices_model.pickle', 'rb'))

# Load the columns from the JSON file
with open('columns.json', 'r') as file:
    model_columns = json.load(file)['data_columns']

st.title('House price Prediction App')

st.divider()

st.write('This app uses ML for predicting House price')

st.divider()

# Input fields
bhk = st.number_input('Number of BHK', min_value=0, value=0)
bathrooms = st.number_input('Number of bathrooms', min_value=0, value=0)
area = st.number_input('Area in sq. ft.', min_value=0, value=2000)

# Create a selectbox with the locations from the model_columns
location = st.selectbox('Choose a location', model_columns[3:])  # Skip the first 3 non-location columns

st.divider()

# Prepare input data for prediction
X = np.zeros(len(model_columns))
X[model_columns.index('bhk')] = bhk
X[model_columns.index('bath')] = bathrooms
X[model_columns.index('total_sqft')] = area

if location in model_columns:
    X[model_columns.index(location)] = 1

# Predict button
predictbutton = st.button('Predict')

if predictbutton:
    st.balloons()
    prediction = model.predict([X])[0]
    st.write(f'Price prediction is {prediction:.2f} INR')
