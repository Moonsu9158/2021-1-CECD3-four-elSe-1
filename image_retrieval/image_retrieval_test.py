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

import os
from glob import glob
from typing_extensions import TypeVarTuple

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir='~/KimHeesu/munsu/image-retrieval/data/train'
test_dir='~/KimHeesu/munsu/image-retrieval/data/test'

#Hyperparameter Tuning
num_epochs=10
batch_size=32
learning_rate=0.001
dropout_rate=0.7
input_shape=(512,512)
num_classes=80

#Preprocess
train_datagen=ImageDataGenerator(
    width_sihft_range=0.3,
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

AutoencoderRetrievalGeneratorModel.fit(train_generator, validation_generator)