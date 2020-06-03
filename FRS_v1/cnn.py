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
np.set_printoptions(threshold=np.inf)
# 定義梯度下降批量
batch_size = 256
# 定義分類數量
num_classes = 4
# 定義訓練週期
epochs = 12

# 定義圖像寬、高
img_rows, img_cols = 64, 64

# 載入  訓練資料
DATADIR = "G:/FRS/trainingdata"

CATEGORIES = ["data01", "data02", "data03","other_faces"]
training_data = []
IMG_SIZE = 64

filename = "G:\\FRS\\x_test.sav"
x_test = pickle.load(open(filename,'rb'))
filename = "G:\\FRS\\x_train.sav"
x_train = pickle.load(open(filename,'rb'))
filename = "G:\\FRS\\y_train.sav"
y_train = pickle.load(open(filename,'rb'))
filename = "G:\\FRS\\y_test.sav"
y_test = pickle.load(open(filename,'rb'))
# 保留原始資料，供 cross tab function 使用
y_test_org = y_test
print(x_test.shape)

# channels_last: 色彩通道(R/G/B)資料(深度)放在第4維度，第2、3維度放置寬與高
# x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

print(x_test.shape)
# 轉換色彩 0~255 資料為 0~1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# y 值轉成 one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 建立簡單的線性執行的模型
model = Sequential()
# 建立卷積層，filter=32,即 output space 的深度, Kernal Size: 3x3, activation function 採用 relu
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# 建立卷積層，filter=64,即 output size, Kernal Size: 3x3, activation function 採用 relu
model.add(Conv2D(64, (3, 3), activation='relu'))
# 建立池化層，池化大小=2x2，取最大值
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
model.add(Dropout(0.25))
# Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
model.add(Flatten())
# 全連接層: 128個output
model.add(Dense(128, activation='relu'))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.5
model.add(Dropout(0.5))
# 使用 softmax activation function，將結果分類
model.add(Dense(num_classes, activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 進行訓練, 訓練過程會存在 train_history 變數中
train_history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# 顯示損失函數、訓練成果(分數)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# save model
filename = "G:\\FRS\\cnn_v4.sav"
pickle.dump(model,open(filename,"wb"))

