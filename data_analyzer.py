
import matplotlib.pyplot as plt
import numpy as np
import sys
import image_augmentation
import data_handler
import scipy.misc
import cv2

class DataAnalyzer:

    def __init__(self):
        print('Init.')


    def showDataDistribution(self, data):
        plt.hist(data, bins = 500)
        plt.show()


if __name__ == '__main__':
    #data_handler = data_handler.DataHandler(data_dir='./velox_data_path', data_description_file='augmented_log.csv', contains_full_path = False)
    #data_handler = data_handler.DataHandler(data_dir='./data', data_description_file='driving_log.csv', contains_full_path=False)


    #data_analyzer = DataAnalyzer()
    #data_analyzer.showDataDistribution(data_handler.get_data_y())

    #image = scipy.misc.imresize(scipy.misc.imread(filename)[25:135], [66, 200])
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    img =  scipy.misc.imread('./velox_data_path/IMG/2013-01-01-01-10-25_img_0.jpg')[220:480]
    print(img.shape)
    cv2.imshow('test', img)
    cv2.waitKey(0)


