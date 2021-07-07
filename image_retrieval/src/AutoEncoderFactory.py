
import keras
from .SimpleAE import SimpleAE
from .StackedAE import StackedAE


class AutoEncoderFactory:
    def makeAE(self,modelName,info):
        if modelName == "simpleAE":
            return SimpleAE(info)
        elif modelName == "convAE":
            pass
        elif modelName == "stackedAE":
            return StackedAE(info)
            

        