import tensorflow as tf
from PIL import Image
import numpy as np

IMAGE_SIZE = (256, 256)

def load_model(path="model.keras"):
    model = tf.keras.models.load_model(path, compile=False)
    model.compile()
    return model

def preprocess_image(image):
    img = image.resize(IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array
