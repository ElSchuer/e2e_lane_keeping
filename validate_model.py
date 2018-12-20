from keras.models import load_model
import seaborn as sns
import numpy as np
import cv2
import matplotlib.pyplot as plt
import cnn_model
import scipy.misc
import data_handler
import csv
import time

class ModelValidator:
    def __init__(self, model_file, val_data_path, vec_spec, desc_file = 'data_labels.csv', show_plot = True, show_image = True, is_full_file_path=False):
        self.times = []

        self.val_data_path = val_data_path
        self.desc_file = desc_file
        self.vec_spec = vec_spec

        self.is_full_file_path = is_full_file_path

        self.show_plot = show_plot
        self.show_image = show_image

        self.model = load_model(model_file)

        self.vec_spec = data_handler.VehicleSpec(angle_norm=30, image_crop_vert=[220, 480])

    def validate_model(self):
        with open(self.val_data_path + '/' + self.desc_file, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=',')

            pred_angles = []
            gt_angles = []

            for row in reader:

                if self.is_full_file_path:
                    imgFile = row[0]
                else:
                    imgFile = data_dir + "/" + row[0]

                gt_angle = float(row[3])

                start_time = time.time()

                image = scipy.misc.imread(imgFile)
                image_resized = scipy.misc.imresize(image[self.vec_spec.image_crop_vert[0]:self.vec_spec.image_crop_vert[1]],
                                                    [cnn_model.input_height, cnn_model.input_width])
                image_resized = np.expand_dims(np.array(image_resized), axis=2)
                image_resized = image_resized[None, :, :, :]

                # Calculate new Steering angle based on image input
                steering_angle = float(self.model.predict(image_resized, batch_size=1))
                steering_angle = steering_angle * vec_spec.angle_norm

                end_time = time.time()
                self.times.append(end_time-start_time)

                pred_angles.append(steering_angle)
                gt_angles.append(gt_angle)

                print("pred_angle = ", steering_angle, " gt_angle = ", gt_angle)

                if self.show_image:
                    self.draw_image(image, steering_angle, gt_angle)

            if self.show_plot:
                self.plot_error(pred_angles, gt_angles)

                #error = np.sqrt(np.power(steering_angle - gt_angle, 2))

        print("Mean Error : " + str(np.mean(np.sqrt(np.power(np.array(pred_angles) - np.array(gt_angles), 2)))))
        print("Mean PredictionTime : " + str(np.mean(self.times)))

    def plot_error(self, pred_angles, gt_angles):

        error_values = np.sqrt(np.power(np.array(pred_angles) - np.array(gt_angles), 2))

        x_values = np.arange(0, len(error_values))
        plt.plot(x_values, error_values, 'C1')
        plt.xlabel('Sample Number')
        plt.ylabel('Steering Angle MSE')
        plt.show()

        plt.plot(x_values, pred_angles)
        plt.plot(x_values, gt_angles)
        plt.xlabel('Sample Number')
        plt.ylabel('Steering Angle')
        plt.show()

    def draw_image(self, image, steering_angle, gt_angle):
        img_height = image.shape[0]
        img_width = image.shape[1]
        ref_line_1 = (int(img_width / 2), img_height)
        ref_line_2 = (int(img_width / 2), int(img_height * 0.5))

        dx = np.tan(np.radians(steering_angle)) * (ref_line_1[1] - ref_line_2[1])
        angle_line_1 = ref_line_1
        angle_line_2 = (int(ref_line_2[0] + dx), ref_line_2[1])

        dx = np.tan(np.radians(gt_angle)) * (ref_line_1[1] - ref_line_2[1])
        gt_line_1 = ref_line_1
        gt_line_2 = (int(ref_line_2[0] + dx), ref_line_2[1])

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        cv2.line(image, ref_line_2, ref_line_1, (255, 255, 255), 1)
        cv2.line(image, angle_line_2, angle_line_1, (0, 0, 255), 2)
        cv2.line(image, gt_line_2, gt_line_1, (0, 255, 0), 2)

        cv2.imshow("Image", image)
        cv2.waitKey(1)

        time.sleep(0.01)

if __name__ == '__main__':
    data_dir = 'C:/Users/lschuermann/Documents/data/images_val'
    desc_file = 'data_labels.csv'
    vec_spec = data_handler.VehicleSpec(angle_norm=30, image_crop_vert=[220, 480])

    model_validator = ModelValidator(model_file='./save/nvidia_model.h5', vec_spec=vec_spec,
                                     val_data_path=data_dir, desc_file=desc_file, show_image=True, show_plot=True)

    model_validator.validate_model()
