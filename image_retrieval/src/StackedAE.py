import keras
import numpy as np

from .AbstractAE import AbstractAE

class StackedAE(AbstractAE):
    def __init__(self, info):
        super(StackedAE, self).__init__(info)

    def makeAutoencoder(self):
        self.input = keras.layers.Input(shape=self.shape_img)

        # encoder
        x = keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu", padding="same")(input)
        x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(x)
        x = keras.layers.Conv2D(128, kernel_size=(3,3), strides=1, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001),padding="same")(x)
        x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(x)
        x = keras.layers.Conv2D(256, kernel_size=(3,3), activation="relu", kernel_regularizer=keras.regularizers.l2(0.001), padding="same")(x)
        x = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(x)
        x = keras.layers.Conv2D(512, kernel_size=(3,3), activation="relu", kernel_regularizer=keras.regularizers.l2(0.001), padding="same")(x)
        self.encoded = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(x)
        
        # decoder
        x = keras.layers.Conv2D(512, kernel_size=(3,3), kernel_regularizer=keras.regularizers.l2(0.001),activation="relu", padding="same")(self.encoded)
        x = keras.layers.UpSampling2D(size=2)(x)
        x = keras.layers.Conv2D(512, kernel_size=(3,3), kernel_regularizer=keras.regularizers.l2(0.001), activation="relu", padding="same")(x)
        x = keras.layers.UpSampling2D(size=2)(x)
        x = keras.layers.Conv2D(256, kernel_size=(3,3), kernel_regularizer=keras.regularizers.l2(0.001), activation="relu", padding="same")(x)
        x = keras.layers.UpSampling2D(size=2)(x)
        x = keras.layers.Conv2D(128, kernel_size=(3,3), kernel_regularizer=keras.regularizers.l2(0.001), activation="relu", padding="same")(x)
        x = keras.layers.UpSampling2D(size=2)(x)
        x = keras.layers.Conv2D(64, kernel_size=(3,3), kernel_regularizer=keras.regularizers.l2(0.001), activation="relu", padding="same")(x)
        x = keras.layers.UpSampling2D(size=2)(x)
        self.decoded = keras.layers.Conv2D(3, kernel_size=(3,3), kernel_regularizer=keras.regularizers.l2(0.001), activation="relu", padding="same")(x)
        
        self.autoencoder = keras.Model(self.input, self.decoded)
        return self.autoencoder

    def makeEncoder(self):
        self.encoder = keras.Model(self.input, self.encoded)
        return self.encoder
    
    def makeDecoder(self):
        output_encoder_shape = self.encoder.layers[-1].output_shape[1:]
        decoded_input = keras.Input(shape=output_encoder_shape)
        decoded_output = self.autoencoder.layers[-11](decoded_input)
        decoded_output = self.autoencoder.layers[-10](decoded_output)
        decoded_output = self.autoencoder.layers[-9](decoded_output)
        decoded_output = self.autoencoder.layers[-8](decoded_output)
        decoded_output = self.autoencoder.layers[-7](decoded_output)
        decoded_output = self.autoencoder.layers[-6](decoded_output)
        decoded_output = self.autoencoder.layers[-5](decoded_output)
        decoded_output = self.autoencoder.layers[-4](decoded_output)
        decoded_output = self.autoencoder.layers[-3](decoded_output)
        decoded_output = self.autoencoder.layers[-2](decoded_output)
        decoded_output = self.autoencoder.layers[-1](decoded_output) 

        self.decoder = keras.Model(decoded_input, decoded_output)
        return self.decoder

        
