import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
import cv2
import math

def crop_and_reshape(image):
    '''
    Crop the top and bottom portion of the image to remove non-interesting region
    Resize the image to reduce training time
    '''
    #crop
    image = image[50:120,:,:]
    #resize
    image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)    
    return image 

def pick_data(data, bins=[-1,-0.5,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.5,1]):
    '''
    Randomly pick a steering bin, then randomly pick a datapoint from the bin
    '''
    categories = pd.cut(data['steering'], bins, labels=bins[1:])
    random_bin_label = bins[1+np.random.randint(len(bins)-2)]
    matching_indices = categories[categories==random_bin_label].index.tolist()
    random_data_index = np.random.randint(len(matching_indices)-1)
    return data.iloc[matching_indices[random_data_index]]   

def preprocess_image_file_train(data_row): 
    '''
    Image preprocessing pipeline for training data with left/right image 
    augmentation and random flipping
    '''
    # pick left, center or right image
    angle_offset = 0
    camera_position = np.random.randint(3)  #center=0, left=1, right=2
    file_path = 'data/'+list(data_row)[camera_position].strip()
    if (camera_position == 1):#left camera. from the left camera's perspective, want to turn the car to the left
        angle_offset = 0.24
    if (camera_position == 2):
        angle_offset = -0.24
    steering = list(data_row)[3] + angle_offset
    
    #randomly flip image
    image = mpimg.imread(file_path)
    if np.random.randint(2)==0:
        image = cv2.flip(image,1)
        steering = -steering

    #crop and reshape
    image = crop_and_reshape(image)
        
    return image,steering

def preprocess_image_file_predict(data_row):
    '''
    Image preprocessing pipeline for validation data that only uses center images
    '''
    file_path = 'data/'+list(data_row)[0].strip()
    image = mpimg.imread(file_path)
    steering = list(data_row)[3]
    image = crop_and_reshape(image)
    return image, steering

def batch_generate_images(data, preprocess, batch_size = 64):
    '''
    The generator to generate batch_size number of images 
    '''
    shape = (batch_size,)+(64,64,3)
    batch_images = np.zeros(shape)
    batch_steering = np.zeros(batch_size)
    while True:
        for i in range(batch_size):
            x,y = preprocess(pick_data(data))
            batch_images[i] = x
            batch_steering[i] = y
        yield batch_images, batch_steering

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.layers import Input
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Cropping2D
from pathlib import Path
import json
import tensorflow as tf

def save_model(fileJson,fileH5):
    print("Saving model and weights: ",fileJson, fileH5)
    if Path(fileJson).is_file():
        os.remove(fileJson)
    json_string = model.to_json()
    with open(fileJson,'w' ) as f:
        json.dump(json_string, f)
    if Path(fileH5).is_file():
        os.remove(fileH5)
    model.save_weights(fileH5)
    
def get_model():
    # activation = PReLU(init='zero', weights=None)
    activation = Activation('relu')

    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64,64,3)))

    # Convlutional layers
    model.add(Convolution2D(24, 5, 5, border_mode='same'))
    model.add(activation)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(36, 5, 5, border_mode='same'))
    model.add(activation)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(48, 5, 5, border_mode='same'))
    model.add(activation)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(activation)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(activation)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(1164))
    model.add(activation)

    model.add(Dense(100))
    model.add(activation)

    model.add(Dense(50))
    model.add(activation)

    model.add(Dense(10))
    model.add(activation)

    model.add(Dense(1))

    model.summary()
    
    return model