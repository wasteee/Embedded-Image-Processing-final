# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:05:52 2020

@author: waster
"""
import pickle
import cv2
import os
from tqdm import tqdm
import random
import numpy as np


IMG_SIZE = 64
X = [] 
y = []
training_data = []

# load cnn model
filename = "G:\\FRS\\cnn_v4.sav"
model = pickle.load(open(filename,'rb'))

#load test picture
path = 'G:\\FRS\\trainingdata\\other_faces\\116.jpg'
img = cv2.imread(path,1)
testing = cv2.imread(path ,cv2.IMREAD_GRAYSCALE)

# translation data structure to (1,64,64,1)
training_data.append([testing, 1])
for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# predict test picture
predictions = model.predict_classes(X)
cv2.imshow("test",img)
print(predictions)

cv2.waitKey(0)
cv2.destroyAllWindows()