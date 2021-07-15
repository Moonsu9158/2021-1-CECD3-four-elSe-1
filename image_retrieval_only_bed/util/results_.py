import cv2
import matplotlib.pyplot as plt
from util.plot_ import plot_

def results_(files, query, result):
    """
    Plotting the N similar images from the dataset with query image.
    Arguments:
    query - (string) - filename of the query image
    result - (list) - filenames of similar images
    """

    def read(img):
        image = cv2.imread('./dataset/'+img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    plt.figure(figsize=(10, 5))
    if type(query) != type(30):
        plot_(query, "", "", 1, 1, 1, "Query Image", "", "", "", True)
    else:
        plot_(read(files[query]), "", "", 1, 1, 1,
              "Query Image "+files[query], "", "", "", True)
    plt.show()
    plt.figure(figsize=(20, 5))
    for iter, i in enumerate(result):
        plot_(read(files[i]), "", "", 1, len(result),
              iter+1, files[i], "", "", "", True)
    plt.show()
