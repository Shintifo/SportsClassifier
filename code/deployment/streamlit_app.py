import streamlit as st
import requests
from PIL import Image



def show_img(file):
    image = Image.open(file)
    st.image(image, use_column_width=True)


def main():
    st.title("Image Sports Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        show_img(uploaded_file)
        if st.button("Predict", use_container_width=True):
            predict(uploaded_file)

def predict(uploaded_file):
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://localhost:8000/predict/", files=files)
    prediction = response.json()["prediction"]

    output = f"""
        <div style="text-align: center; font-size: 24px;">
            Prediction: <b>{prediction}</b>
        </div>
    """

    st.markdown(output, unsafe_allow_html=True)


if __name__ == '__main__':
    main()