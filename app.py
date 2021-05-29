
########################## Imports #############################################
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model, load_model
import time
import codecs
import collections
from six.moves import cPickle
import json
import numpy as np

############################ Loaders ############################################

app = Flask('AgVenture',template_folder='template')

model = load_model("model/mymodel.h5py")

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

# Predict function

# @app.route('/upload')
# def upload_file():
#    return render_template('home.html')
    
@app.route('/upload_file', methods = ['GET','POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        files=f.read()
        output=predictions(files)
    return render_template('end.html',generated=output)


@app.route('/',methods = ['GET'])
def ping():
    
    return render_template('home.html')

# Main

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)