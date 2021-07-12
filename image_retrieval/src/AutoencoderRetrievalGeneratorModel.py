from image_retrieval.src.AutoencoderRetrievalModel import AutoencoderRetrievalModel
import os
from glob import glob

import keras
import numpy as np
from utils import split
from keras.callbacks.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .AbstractRetrievalModel import AbstractRetrievalModel


layers=keras.layers

class AutoencoderRetrievalGeneratorModel(AbstractRetrievalModel):
    def __init__(self, modelName, info):
        super(AutoencoderRetrievalModel, self),__init__(modelName, info)
        self.autoencoder=None
        self.encoder=None
        self.decoder=None
    
    def getShape_img(self):
        return self.ae.getShape_img()

    def getInputshape(self):
        return self.ae.getInputshape()

    def getOutputshape(self):
        return self.ae.getOutputshape()

    def fit(train_generator, validation_generator, n_epochs=50, batch_size=256, callbacks=None):
        indices_fracs=split(fracs=[0.9, 0.1], N=len(X), seed=0)
        self.autoencoder.fit_generator(train_generator,
                                       steps_per_epoch=len(train_generator),
                                       epochs=n_epochs,
                                       validation_data=validation_generator,
                                       validation_steps=len(validation_generator))

    def predict(self, X):
        return self.encoder.predict(X)

    def set_arch(self):
        autoencoderFactory=RetrievalModelFactory()
        self.ae=atoencoderFactory.makeRetrievalModel(self.modelName, self.info)

        self.autoencoder=self.ae.makeAutoencoder()
        self.encoder=self.ae.makeEncoder()
        self.decoder=self.ae.makeDecoer()

        print("\nautoencoder.summary():")
        print(self.autoencoder.summary())
        print("\nencoder.summary():")
        print(self.encoder.summary())
        print("\ndecoder.summary():")
        print(self.decoder.summary())

    def compile(self, loss="binary_crossentropy", optimizer="adam"):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    def load_models(self, loss="binary_crossentropy", optimizer="adam"):
        print("Loading models...")
        self.autoencoder=keras.models.load_model(
            self.info["autoencoderFile"]
        )
        self.encoder=keras.models.load_model(self.info["encoderFile"])
        self.decoder=keras.models.load_model(self.info["decoderFile"])
        self.autoencoder.compile(optimizer=optimizer,loss=loss)
        self.encoder.compile(optimizer=optimizer,loss=loss)
        self.decoder.compile(optimizer=optimizer, loss=loss)

    def save_models(self):
        print("Saving models...")
        self.autoencoder.save(self.info["autoencoderFile"])
        self.encoder.save(self.info["encoderFile"])