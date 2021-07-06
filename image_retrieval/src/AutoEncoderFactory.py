
import keras
from . import SimpleAE
from . import StackedAE


class AutoEncoderFactory:
    def makeAE(self,modelName,info):
        if modelName == "simpleAE":
            ae = SimpleAE(info)
        elif modelName == "convAE":
            pass
        elif modelName == "stackedAE":
            ae = StackedAE(info)
            

        