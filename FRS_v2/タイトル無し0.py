import pickle
import cv2
import os
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import timeit
IMG_SIZE = 64
input_dir = './trainingdata/test'


# load cnn model
filename = "G:\\FRS\\cnn_v9.sav"
model = pickle.load(open(filename,'rb'))

#load test picture
count = 0
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            X = [] 
            y = []
            training_data = []
            img_path = path+'/'+filename
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



            # translation data structure to (1,64,64,1)
            training_data.append([gray_img, 1])
            for features,label in training_data:
                X.append(features)
                y.append(label)
            
            X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
            start = timeit.default_timer()
            # predict test picture
            predictions = model.predict_classes(X)
            
            if(predictions != 5):
                count +=1
                plt.hist(X.ravel(), 255, [0, 256])
                plt.show()
                print(img_path,predictions)
            stop = timeit.default_timer()
            t1 = stop - start
            print(t1,'s')
print(count)