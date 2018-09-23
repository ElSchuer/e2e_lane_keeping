import scipy
import csv
import cv2
import numpy as np
from random import shuffle
import scipy.misc

class VehicleSpec:

    def __init__(self, angle_norm, image_crop_vert):
        self.angle_norm = angle_norm
        self.image_crop_vert = image_crop_vert

class DataHandler:

    def __init__(self,data_dir,  data_description_file, vehicle_spec ,contains_full_path = False, convert_image = True, image_channels=3):
        self.data_desc_file = data_description_file
        self.data_dir = data_dir
        self.is_full_file_path = contains_full_path
        self.vehicle_spec = vehicle_spec
        self.convert_image = convert_image
        self.image_channels = image_channels

        self.data = self.get_meta_data_from_file(self.data_desc_file)
        shuffle(self.data)

        print("Complete Data : " + str(len(self.data)))

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

    def add_axis(self, data):
        return np.array(data).reshape(
            [np.array(data).shape[0],
             np.array(data).shape[1], 1])

    def get_meta_data_from_file(self, data_desc_file):
        data = []
        with open(self.data_dir + '/' + self.data_desc_file, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=',')

            for row in reader:

                if self.is_full_file_path:
                    image = self.get_image(row[0])
                else:
                    image = self.get_image(self.data_dir + '/' + row[0])

                if self.image_channels == 1:
                    image = self.add_axis(image)

                angle = float(row[3])/self.vehicle_spec.angle_norm
                data.append([image, angle])


        return data

    def get_image(self, filename):
        image = scipy.misc.imresize(scipy.misc.imread(filename)[self.vehicle_spec.image_crop_vert[0]:self.vehicle_spec.image_crop_vert[1]], [66, 200])

        if self.convert_image:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

        return (image / 255.0)

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

            print("Train Data Samples : " + str(len(self.train_data)))
            print("Val Data Samples : " + str(len(self.val_data)))
        else:
            print("Invalid validation split. Split has to be < 1")

        return self.train_data, self.val_data

