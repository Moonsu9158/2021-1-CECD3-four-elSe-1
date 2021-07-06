
import keras
from src.utils import split
import tensorflow as tf
import numpy as np
from .AutoEncoderFactory import AutoEncoderFactory

"""

 autoencoder.py  (author: Anson Wong / git: ankonzoid)

"""
layers = keras.layers

class AutoEncoder:

    def __init__(self, modelName, info):
        self.modelName = modelName
        self.info = info
        self.autoencoder = None
        self.encoder = None
        self.decoder = None

    # Train
    def fit(self, X, n_epochs=50, batch_size=256):
        indices_fracs = split(fracs=[0.9, 0.1], N=len(X), seed=0)
        X_train, X_valid = X[indices_fracs[0]], X[indices_fracs[1]]
        self.autoencoder.fit(X_train, X_train,
                             epochs=n_epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(X_valid, X_valid))

    # Inference
    def predict(self, X):
        return self.encoder.predict(X)

    # Set neural network architecture
    def set_arch(self):
        autoencoderFactory = AutoEncoderFactory()
        ae = autoencoderFactory.makeAE(self.modelName,self.info)

        self.autoencoder = ae.makeAutoencoder()
        self.encoder = ae.makeEncoder()
        self.decoder = ae.makeDecoder()

        # Generate summaries
        print("\nautoencoder.summary():")
        print(self.autoencoder.summary())
        print("\nencoder.summary():")
        print(self.encoder.summary())
        print("\ndecoder.summary():")
        print(self.decoder.summary())


    # Compile
    def compile(self, loss="binary_crossentropy", optimizer="adam"):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    # Load model architecture and weights
    def load_models(self, loss="binary_crossentropy", optimizer="adam"):
        print("Loading models...")
        self.autoencoder = keras.models.load_model(
            self.info["autoencoderFile"])
        self.encoder = keras.models.load_model(self.info["encoderFile"])
        self.decoder = keras.models.load_model(self.info["decoderFile"])
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
        self.encoder.compile(optimizer=optimizer, loss=loss)
        self.decoder.compile(optimizer=optimizer, loss=loss)

    # Save model architecture and weights to file
    def save_models(self):
        print("Saving models...")
        self.autoencoder.save(self.info["autoencoderFile"])
        self.encoder.save(self.info["encoderFile"])
