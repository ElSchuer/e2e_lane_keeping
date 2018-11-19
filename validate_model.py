import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import cnn_model
import scipy.misc
import data_handler
import csv

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, './save/car_model.ckpt')

    vec_spec = data_handler.VehicleSpec(angle_norm=30, image_crop_vert=[220, 480])

    data_dir = "./data/velox_data_augmented"
    desc_file = 'data_labels.csv'
    is_full_file_path = False


    with open(data_dir + '/' + desc_file, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:

            if is_full_file_path:
                imgFile = row[0]
            else:
                imgFile = data_dir + "/" + row[0]

            gt_angle = float(row[3])/vec_spec.angle_norm

            image = scipy.misc.imread(imgFile)
            image_resized = scipy.misc.imresize(image[vec_spec.image_crop_vert[0]:vec_spec.image_crop_vert[1]],[66, 200]) / 255.0 - 0.5
            image_resized = image_resized.reshape([np.array(image_resized).shape[0], np.array(image_resized).shape[1], 1])

            # Calculate new Steering angle based on image input
            steering_angle = cnn_model.y.eval(session=sess, feed_dict={cnn_model.x: image_resized[None, :, :], cnn_model.keep_prob: 1.0})[0][0]
            steering_angle = steering_angle * vec_spec.angle_norm

            img_height = image.shape[0]
            img_width = image.shape[1]
            ref_line_1 = (int(img_width / 2), img_height)
            ref_line_2 = (int(img_width / 2), int(img_height*0.5))
            print(ref_line_1[1] - ref_line_2[1])
            dx = np.tan(np.radians(steering_angle)) * (ref_line_1[1] - ref_line_2[1])

            print(dx)

            angle_line_1 = ref_line_1
            angle_line_2 = (int(ref_line_2[0] + dx), ref_line_2[1])

            cv2.line(image, ref_line_2, ref_line_1, (255, 0, 0), 5)
            cv2.line(image, angle_line_2, angle_line_1, (255, 0, 0), 5)

            print(steering_angle)

            cv2.imshow("TEST", image)
            cv2.waitKey(0)