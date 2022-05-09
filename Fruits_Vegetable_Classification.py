import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import sys
from streamlit import cli as stcli

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow import keras
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
import webbrowser

model = load_model('model.h5')

url = "http://localhost:8502/"

labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']

vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

prices = {'Apple': 52, 'Banana': 21, 'Bello Pepper': 13, 'Chilli Pepper': 33, 'Grapes': 39, 'Jalepeno': 12.5,
          'Kiwi': 38, 'Lemon': 34, 'Mango': 199, 'Orange': 50,
          'Paprika': 15, 'Pear': 30, 'Pineapple': 40, 'Pomegranate': 35, 'Watermelon': 15, 'Beetroot': 20,
          'Cabbage': 25, 'Capsicum': 60, 'Carrot': 75, 'Cauliflower': 25, 'Corn': 40, 'Cucumber': 30, 'Eggplant': 60,
          'Ginger': 35,
          'Lettuce': 45, 'Onion': 20, 'Peas': 35, 'Potato': 15, 'Raddish': 30, 'Soy Beans': 40, 'Spinach': 20,
          'Sweetcorn': 20, 'Sweetpotato': 30,
          'Tomato': 20, 'Turnip': 40}


def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()


def fetch_prices(prediction):
    price = prices[prediction]
    return price


def run():
    st.title("Automated Self Checkout System 🍍 🍅 ")
    a = st.radio("Enter image input ", ['Web-cam', 'File-upload'], 1)
    if a == "Web-cam":
        img_file = st.camera_input("Take a picture")
    if a == "File-upload":
        img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])

    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        result = processed_img(save_image_path)
        print(result)
        if result in vegetables:
            st.info('**Category : Vegetables**')
            st.success("**Predicted : " + result + '**')
            pl = fetch_prices(result)
            pl = f" ₹ {pl}"
            st.metric(label=result, value=pl)

        else:
            st.info('**Category : Fruit**')
            st.success("**Predicted : " + result + '**')
            pl = fetch_prices(result)
            pl = f" ₹ {pl}"
            st.metric(label=result, value=pl)


run()
