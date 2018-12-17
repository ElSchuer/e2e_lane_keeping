import scipy
import csv
import cv2
import numpy as np
import scipy.misc
import cnn_model

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

        self.x_data = []
        self.y_data = []

    def read_data(self):
        self.x_data, self.y_data = self.get_data()

        print("Complete Data : " + str(len(self.x_data)))

    def add_axis(self, data):
        return np.array(data).reshape(
            [np.array(data).shape[0],
             np.array(data).shape[1], 1])


    def get_data(self):
        images = []
        angles = []
        sample_count = 0
        print('Reading data from path ' + self.data_path)
        with open(self.data_path + '/' + self.desc_file, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=',')

            for row in reader:

                if len(row) <= 0:
                    continue

                if self.is_full_file_path:
                    image = self.get_image(row[0])
                else:
                    image = self.get_image(self.data_dir + '/' + row[0])

                angle = float(row[3]) / self.vehicle_spec.angle_norm

                images.append(image)
                angles.append(angle)
                sample_count = sample_count + 1

    def get_data(self):
        images = []
        angles = []
        print('Reading data from path ' + self.data_dir)
        with open(self.data_dir + '/' + self.data_desc_file, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=',')

            for row in reader:

                if len(row) <= 0:
                    continue

                if self.is_full_file_path:
                    image = self.get_image(row[0])
                else:
                    image = self.get_image(self.data_dir + '/' + row[0])

                if self.image_channels == 1:
                    image = self.add_axis(image)

                angle = float(row[3])/self.vehicle_spec.angle_norm

                images.append(image)
                angles.append(angle)

        return images,angles


    def get_image(self, filename):
        image = scipy.misc.imresize(scipy.misc.imread(filename)[
                                    self.vehicle_spec.image_crop_vert[0]:self.vehicle_spec.image_crop_vert[1]],
                                    [cnn_model.input_height, cnn_model.input_width])

        if self.convert_image:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

        return image
