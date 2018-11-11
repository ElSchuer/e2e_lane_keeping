
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
        plt.hist(data, bins = 300)
        plt.show()

    def print_samples_not_equal_zero(self, data):
        print(str(np.count_nonzero(np.array(data))  * 100 / len(data)) + ' % of the samples are not equal zero.')






