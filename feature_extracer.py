#!/usr/bin/env python
# coding: utf-8


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np


# Get the image as input and return as deep feature
class FeatureExtracer:
    def __init__(self):
        base_model = VGG16(weights="imagenet")
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)

   ## def extract(self, img):
   ##     return None
    
    def extract(self, img):
        img = img.resize((224, 224)).convert("RGB") # RGB should be for Python
        x = image.img_to_array(img) # to np.array
        x = np.expand_dims(x, axis=0) # (H, W, C) -> (1, H, W, C)
        x = preprocess_input(x) # Subtract avg pixel value
        feature = self.model.predict(x)[0] # (1, 4096) -> (4096)
        return feature / np.linalg.norm(feature) # Normalize



