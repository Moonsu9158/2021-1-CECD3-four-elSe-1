from abc import *

class AbstractAE(metaclass=ABCMeta):
    def __init__(self, info):
        self.shape_img = info["shape_img"]

    @abstractmethod
    def makeAutoencoder(self):
        pass
    
    @abstractmethod
    def makeEncoder(self):
        pass

    @abstractmethod
    def makeDecoder(self):
        pass
