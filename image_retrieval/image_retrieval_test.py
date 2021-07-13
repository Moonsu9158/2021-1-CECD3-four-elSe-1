"""
from image_retrieval import image_retrieval

import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
dataset_path = "./image-retrieval/data/train"
images_path = glob(dataset_path+"*.jpg")
for path in tqdm(images_path):
    image_pil=Image.open(path)
    image_resized=image_pil.resize((512,512))
    image_resized.save(path)

image_retrieval()
"""

from image_retrieval.src.SimpleAE import SimpleAE
import os
from glob import glob

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir=os.path.join(os.getcwd(), "data", "train")
test_dir=os.path.join(os.getcwd(), "data", "test")

#Hyperparameter Tuning
num_epochs=10
batch_size=32
learning_rate=0.001
dropout_rate=0.7
input_shape=(28,28,3)
num_classes=80

#Preprocess
train_datagen=ImageDataGenerator(
    width_shift_range=0.3,
    zoom_range=0.2,
    rescale=1./255.,
    horizontal_flip=True
)

test_datagen=ImageDataGenerator(
    rescale=1./255.
)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical'
)

validation_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical'
)

"""
inputs=layers.Input(input_shape)
net = layers.Conv2D(32, (3,3), padding="SAME")(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3,3), padding="SAME")(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2,2))(net)
net = layers.Dropout(dropout_rate)(net)

net = layers.Conv2D(64, (3,3), padding="SAME")(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3,3), padding="SAME")(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2,2))(net)
net = layers.Dropout(dropout_rate)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(dropout_rate)(net)
net = layers.Dense(num_classes)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')
"""
model=SimpleAE()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch = len(train_generator),
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)