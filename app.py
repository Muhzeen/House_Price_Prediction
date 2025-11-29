import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("🏠 House Price Prediction App")

sqft_living = st.number_input("Living Area (sq ft)", 200, 10000)
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)
floors = st.number_input("Floors", 1, 4)

if st.button("Predict"):
    input_data = np.array([[sqft_living, bedrooms, bathrooms, floors]])
    pred = model.predict(input_data)[0]
    st.success(f"Predicted Price: ₹{round(pred,2)}")
