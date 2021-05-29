########################## Imports #############################################
import os
import glob
import numpy as np
import pandas as pd
from pprint import pprint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Model, layers, Sequential, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping
import matplotlib.pyplot as plt


######################### Functions ############################################
def get_images(path):
	images = []
	for root, dirs, files in os.walk(path):
    	for file in files:
       		images.append(os.path.join(root, file))
    return images

def data_generator(path):
	datagen=ImageDataGenerator(rescale=1/255)
	generator=datagen.flow_from_directionary(
		path,
		class_mode = 'categorical',
    	target_size = (150, 150),
    	color_mode="rgb",
    	shuffle=True
		)
	return generator

def predictions(img_path):
	test_img=img_path
	image=tf.keras.preprocessing.image.load_img(
    test_img, grayscale=False, color_mode="rgb", target_size=(150,150), interpolation="nearest"
	)
	input_arr = tf.keras.preprocessing.image.img_to_array(image)
	input_arr = np.array([input_arr])
	pred=model.predict_generator(input_arr)
	predicted_class_indices=np.argmax(pred,axis=1)
	if (predicted_class_indices==[1]):
  		result='Healthy'
	else:
  		result='Infected'
  	return result



