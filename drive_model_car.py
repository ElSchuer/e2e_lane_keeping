import numpy as np
import scipy.misc
import rospy
import tensorflow as tf
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
import time
import vehicle_spec

class AutonomousModelCarControl:

    def __init__(self, vehicle_spec):
        self.vehicle_spec = vehicle_spec

        self.input_width = 200
        self.input_height = 66

        self.model = self.get_model()
        self.model.load_weights('./save/nvidia_model.h5')
        self.model.summary()

        self.steeringPub = rospy.Publisher("/vehicle_control/steering_angle", Float32, queue_size=1)
        self.sub = rospy.Subscriber("/camera/image_raw", Image, self.predict_steering_angle, queue_size=1, buff_size=2**24)
        self.bridge = CvBridge()


    def get_model(self):
        input_size = (self.input_height, self.input_width, 1)

        dense_keep_prob = 0.8
        init = 'glorot_uniform'

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_size))

        model.add(tf.keras.layers.Conv2D(24, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init,
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Conv2D(36, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init,
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Conv2D(48, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init,
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', strides=(1, 1), kernel_initializer=init,
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', strides=(1, 1), kernel_initializer=init,
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=1164, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(rate=dense_keep_prob))

        model.add(tf.keras.layers.Dense(units=100, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dense(units=50, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dense(units=10, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dense(units=1))

        return model

    def predict_steering_angle(self, image_msg):

        start_time = time.time()

        try:
            image = np.asarray(self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough'))
        except CvBridgeError as e:
            print(e)

        image = scipy.misc.imresize(image[self.vehicle_spec.image_crop_vert[0]:self.vehicle_spec.image_crop_vert[1]],[self.input_height, self.input_width])
        image_resized = np.expand_dims(np.array(image), axis=2)
        image_resized = image_resized[None, :, :, :]

        # Calculate new Steering angle based on image input
        steering_angle = float(self.model.predict(image_resized, batch_size=1))
        steering_angle = steering_angle * vec_spec.angle_norm

        msg = Float32()
        msg.data = steering_angle

        self.steeringPub.publish(msg)



if __name__ == '__main__':
    vec_spec = vehicle_spec.VehicleSpec(angle_norm=30, image_crop_vert=[220, 480])

    car_control = AutonomousModelCarControl(vehicle_spec = vec_spec)

    rospy.init_node('drive_model_car', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
