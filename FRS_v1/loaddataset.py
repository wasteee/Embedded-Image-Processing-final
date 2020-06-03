# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:15:55 2020

@author: waster
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pickle
import os
import cv2
from tqdm import tqdm
import random
import numpy as np
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# define numbers of classes
num_classes = 4
# define image size
img_rows, img_cols = 64, 64

# load training data
DATADIR = "G:/FRS/trainingdata"

CATEGORIES = ["data01", "data02", "data03","other_faces"]
training_data = []
IMG_SIZE = 64
def create_training_data():
    for category in CATEGORIES:  # data01 data02 data03 other_faces

        path = os.path.join(DATADIR,category)  # create path to each dir
        class_num = CATEGORIES.index(category)  # get the classification  (0,1,2,3). 

        for img in tqdm(os.listdir(path)):  # iterate over each image 
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            training_data.append([new_array, class_num])  # add this to our training_data
            

create_training_data()

X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)
print(X[0].shape)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(X.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 41)


#save training data split
filename = "G:\\FRS\\x_train.sav"
pickle.dump(x_train,open(filename,"wb"))
filename = "G:\\FRS\\x_test.sav"
pickle.dump(x_test,open(filename,"wb"))
filename = "G:\\FRS\\y_train.sav"
pickle.dump(y_train,open(filename,"wb"))
filename = "G:\\FRS\\y_test.sav"
pickle.dump(y_test,open(filename,"wb"))

