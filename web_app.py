# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 14:24:54 2023

@author: dell1
"""

import tensorflow as tf
import numpy as np
import streamlit as st
import pandas as pd
from io import StringIO
from keras.preprocessing import image
from PIL import Image, ImageOps


loaded_model = tf.keras.models.load_model('Model/cnn.hdf5')


def pred(input_image):
    z =ImageOps.fit(input_image,(64,64))
    z=image.img_to_array(z)
    z=np.expand_dims(z,axis=0)
    x = loaded_model.predict(z/255.0)
    if x[0][0] > 0.5:
        return 'Dog'
    else:
        return 'Cat'
        
def main():
    st.title('Cat or Dog Prediction')
    uploaded_file = st.file_uploader("Choose a file:")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image,caption='Uploaded File',use_column_width=True)
        
        predict = ' '
        
        if st.button('Predict'):
            predict = pred(image)
            
        st.success(predict)
        
if __name__ == '__main__':
    main()  