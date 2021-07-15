import cv2
import csv
import tensorflow as tf
import imageio as io
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings
from tensorflow import keras
from keras import backend as K
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Activation, MaxPooling2D, UpSampling2D
from keras.models import load_model, Sequential
from keras.optimizers import Adam, Adagrad, RMSprop
from keras.preprocessing import image
from pylab import *
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from glob import glob
from model import encoder_decoder_model
from util.encoder_decoder_model import encoder_decoder_model
from util.image2array import image2array
from util.plot_ import plot_
from util.results_ import results_
from tqdm import tqdm_notebook
from PIL import Image


# bed_image_retrieval/수납 침대/*.png
image_dir = os.path.join(os.getcwd(), "수납 침대", "*.png")  # 사진이 저장되어 있는 위치
image_paths = glob(image_dir)  # 사진만 추출

path = image_paths[0]

gfile = tf.io.read_file(path)
image = tf.io.decode_image(gfile, dtype=tf.float32)

image.shape  # heigth, width, channel 출력 #TensorShape([550, 550, 3])

print("total image len : ", len(image_paths))  # 679
train_data, test_data = train_test_split(
    image_paths, test_size=0.15, random_state=34)
# 데이터 나누기. random_state를 설정해야 매번 데이터셋이 변경되는 것을 방지할 수 있습니다.
print("train_data num : ", len(train_data))  # 577
print("test_data num : ", len(test_data))  # 102

# 나중에 사진 확인을 위해 주소값을 리스트에 저장
train_list = []
test_list = []
for i in range(len(train_data)):
    train_list.append(train_data)

for i in range(len(test_data)):
    test_list.append(test_data)

# csv파일로 변환
with open('train_data_file.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(train_list)
f.close()

with open('test_data_file.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(test_list)
f.close()


heights = []
widths = []

for path in tqdm_notebook(image_paths):
    img_pil = Image.open(path)
    image = np.array(img_pil)
    h = image.shape[0]
    w = image.shape[1]
    heights.append(h)
    widths.append(w)

np.unique(heights)  # array([550])
np.unique(widths)  # array([550])


# load model
model = encoder_decoder_model()
model.summary()
print("\n")

strategy = tf.distribute.MirroredStrategy()

optimizer = Adam(learning_rate=0.001)
with strategy.scope():
    model = encoder_decoder_model()
    model.compile(optimizer=optimizer, loss='mse')
early_stopping = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=6, min_delta=0.0001)
checkpoint = ModelCheckpoint(
    './encoder_model.h5', monitor='val_loss', mode='min', save_best_only=True)

model.fit(train_data, train_data, epochs=35, batch_size=32, validation_data=(
    test_data, test_data), callbacks=[early_stopping, checkpoint])

loss = model.history.history["loss"]
val_loss = model.history.history["val_loss"]

# model.history.history
plt.figure(figsize=(15, 5))
epochs = [i for i in range(34)]
plot_(epochs, loss, '', 1, 2, 1, 'Training loss on each epoch',
      'Epoch', 'Loss', 'training', False, 'g')
plot_(epochs, val_loss, '', 1, 2, 2, 'validation loss on each epoch',
      'Epoch', 'Loss', 'testing', False, 'r')
