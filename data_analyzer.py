
import matplotlib.pyplot as plt
import numpy as np
import sys
import image_augmentation
import data_handler

class DataAnalyzer:

    def __init__(self):
        print('Init.')


    def showDataDistribution(self, data):
        plt.hist(data, bins = 500)
        plt.show()


if __name__ == '__main__':
    data_handler = data_handler.DataHandler(data_dir='./data/augmented_data', data_description_file='augmented_log.csv', contains_full_path = True)

    data_analyzer = DataAnalyzer()
    data_analyzer.showDataDistribution(data_handler.get_data_y())


