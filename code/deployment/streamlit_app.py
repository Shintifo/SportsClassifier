import streamlit as st
import requests


st.title("Image Sports Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://localhost:8000/predict/", files=files)
    prediction = response.json()["prediction"]
    st.write(f"Prediction: {prediction}")
