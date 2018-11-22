import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import cnn_model
import scipy.misc
import data_handler
import csv

class ModelValidator:
    def __init__(self, model_file, val_data_path, vec_spec, desc_file = 'data_labels.csv', plot_error_values = True, show_image = True, is_full_file_path=False):
        self.error_values = []
        self.pos_errors = []
        self.neg_errors = []

        self.val_data_path = val_data_path
        self.desc_file = desc_file
        self.vec_spec = vec_spec

        self.is_full_file_path = is_full_file_path

        self.plot_error_values = plot_error_values
        self.show_image = show_image

        self.sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_file)

        self.vec_spec = data_handler.VehicleSpec(angle_norm=30, image_crop_vert=[220, 480])

        plt.ion()
        plt.show()

    def validate_model(self):
        with open(self.val_data_path + '/' + self.desc_file, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=',')
            for row in reader:

                if self.is_full_file_path:
                    imgFile = row[0]
                else:
                    imgFile = data_dir + "/" + row[0]

                gt_angle = float(row[3])

                image = scipy.misc.imread(imgFile)
                image_resized = scipy.misc.imresize(image[self.vec_spec.image_crop_vert[0]:self.vec_spec.image_crop_vert[1]],
                                                    [66, 200]) / 255.0 - 0.5
                image_resized = image_resized.reshape(
                    [np.array(image_resized).shape[0], np.array(image_resized).shape[1], 1])

                # Calculate new Steering angle based on image input
                steering_angle = cnn_model.y.eval(session=self.sess, feed_dict={cnn_model.x: image_resized[None, :, :],
                                                                           cnn_model.keep_prob: 1.0})[0][0]
                steering_angle = steering_angle * vec_spec.angle_norm

                print("pred_angle = ", steering_angle, " gt_angle = ", gt_angle)

                if self.show_image:
                    self.draw_image(image, steering_angle, gt_angle)

                if self.plot_error_values:
                    self.plot_error(steering_angle, gt_angle)

                error = np.sqrt(np.power(steering_angle - gt_angle, 2))

                if np.sign(gt_angle) < 0:
                    self.neg_errors.append(error)
                elif np.sign(gt_angle) > 0:
                    self.pos_errors.append(error)

                self.error_values.append(error)

        print("Mean Error : " + str(np.mean(self.error_values)))
        print("Mean Neg Error : " + str(np.mean(self.neg_errors)))
        print("Mean Pos Error : " + str(np.mean(self.pos_errors)))

    def plot_error(self, steering_angle, gt_angle):
        error = np.sqrt(np.power(steering_angle - gt_angle, 2))

        if np.sign(gt_angle) < 0:
            self.neg_errors.append(error)
        elif np.sign(gt_angle) > 0:
            self.pos_errors.append(error)

        self.error_values.append(error)

        # Plotting the error values
        if self.plot_error_values:
            x_values = np.arange(0, len(self.error_values))
            plt.plot(x_values, self.error_values, 'C1')
            plt.xlabel('Sample Number')
            plt.ylabel('Steering Angle MSE')
            plt.draw()
            plt.pause(00000.1)

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

if __name__ == '__main__':
    data_dir = "C:/Users/lschuermann/Documents/data/images_val"
    desc_file = 'data_labels.csv'
    vec_spec = data_handler.VehicleSpec(angle_norm=30, image_crop_vert=[220, 480])

    model_validator = ModelValidator(model_file='./save/car_model_3.ckpt', vec_spec=vec_spec,
                                     val_data_path=data_dir, desc_file=desc_file, show_image=True, plot_error_values=False)

    model_validator.validate_model()
