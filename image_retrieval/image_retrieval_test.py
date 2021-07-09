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