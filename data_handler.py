import scipy
import csv
import cv2
import numpy as np
from random import shuffle
import scipy.misc

class DataHandler:

    def __init__(self,data_dir,  data_description_file):
        self.data_desc_file = data_description_file
        self.data_dir = data_dir

        self.data = self.get_meta_data_from_file(self.data_desc_file)

        self.train_data = []
        self.val_data = []

        self.train_iterations = 0
        self.val_iterations = 0


    def get_data_x(self):
        x_data = []
        for i in range(len(self.data)):
            x_data.append(self.data[i][0])

        return np.array(x_data)

    def get_data_y(self):
        y_data = []
        for i in range(len(self.data)):
            y_data.append(float(self.data[i][1]))

        return np.array(y_data)

    def get_meta_data_from_file(self, data_desc_file):
        data = []
        with open(self.data_dir + '/' + self.data_desc_file, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=',')

            rowNum = 0
            for row in reader:

                if rowNum == 0:
                    header = row
                else:
                    image = self.get_image(self.data_dir + '/' + row[0])
                    angle = row[3]
                    data.append([image, angle])
                rowNum = rowNum + 1

        return data

    def get_image(self, filename):
        image = scipy.misc.imresize(scipy.misc.imread(filename)[25:135], [66, 200])
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

        #return (image / 255.0)

        return image

    def get_val_batch(self, batch_size):

        batch_x = []
        batch_y = []

        start_index = self.val_iterations*batch_size
        end_index = (self.val_iterations+1)*batch_size

        if end_index > len(self.val_data):
            end_index = len(self.val_data)

            self.val_iterations = 0
            shuffle(self.val_data)

        for i in range(start_index, end_index):
            batch_x.append(self.val_data[i][0])
            batch_y.append(float(self.val_data[i][1]))

        self.val_iterations = self.val_iterations + 1

        return batch_x, batch_y

    def get_train_batch(self, batch_size):

        batch_x = []
        batch_y = []

        start_index = self.train_iterations*batch_size
        end_index = (self.train_iterations+1)*batch_size

        if end_index > len(self.train_data):
            end_index = len(self.train_data)

            self.train_iterations = 0
            shuffle(self.train_data)

        for i in range(start_index, end_index):
            batch_x.append(self.train_data[i][0])
            batch_y.append(float(self.train_data[i][1]))

        self.train_iterations = self.train_iterations + 1

        return batch_x, batch_y


    def generate_data_splits(self, val_split):
        if val_split < 1.0:
            self.train_data = self.data[0:int((1 - val_split) * len(self.data))]
            self.val_data = self.data[int((1 - val_split) * len(self.data)):]
        else:
            print("Invalid validation split. Split has to be < 1")

        return self.train_data, self.val_data

