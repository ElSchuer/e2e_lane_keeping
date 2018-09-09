
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../image_augmentation/')
from image_augmentation import flip_horizontal
from image_augmentation import manipulate_brightness
import data_handler

class DataAnalyzer:

    def __init__(self):
        print('Init.')


    def showDataDistribution(self, data):
        plt.hist(data, bins = 500)
        plt.show()


if __name__ == '__main__':
    data_handler = data_handler.DataHandler(data_dir='./data', data_description_file='driving_log.csv')

    #data_analyzer = DataAnalyzer()
    #data_analyzer.showDataDistribution(data_handler.get_data_y())
    img_id = 300
    x = data_handler.get_data_x()
    y = data_handler.get_data_y()

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(x[img_id])

    x_flip = manipulate_brightness(x[img_id])
    axarr[1].imshow(x_flip)
    plt.show()

