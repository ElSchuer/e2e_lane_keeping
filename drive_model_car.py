import numpy as np
import data_handler
import cnn_model
import scipy.misc
import rospy
import tensorflow as tf
from sensor_msgs.msg import Image
from messages.msg import CarControlMessage
from cv_bridge import CvBridge, CvBridgeError

class AutonomousModelCarControl:

    def __init__(self, vehicle_spec):
        self.vehicle_spec = vehicle_spec

        self.sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(self.sess, 'save/velox_model.ckpt')


        self.steeringPub = rospy.Publisher("/control/steering_predicted", CarControlMessage, queue_size=1)
        self.sub = rospy.Subscriber("/camera/image_raw", Image, self.predict_steering_angle)
        self.bridge = CvBridge()

    def predict_steering_angle(self, image_msg):

        try:
            image = np.asarray(self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough'))
        except CvBridgeError as e:
            print(e)

        image = scipy.misc.imresize(image[self.vehicle_spec.image_crop_vert[0]:self.vehicle_spec.image_crop_vert[1]],[66, 200])/255.0-0.5
        image = image.reshape([np.array(image).shape[0], np.array(image).shape[1], 1])

        #Calculate new Steering angle based on image input
        steering_angle = cnn_model.y.eval(session = self.sess, feed_dict={cnn_model.x: image[None, :, :], cnn_model.keep_prob: 1.0})[0][0]

        steering_angle = steering_angle*self.vehicle_spec.angle_norm

        msg = CarControlMessage()
        msg.steeringAngle = steering_angle

        self.steeringPub.publish(msg)



if __name__ == '__main__':
    vec_spec = data_handler.VehicleSpec(angle_norm=30, image_crop_vert=[220, 480])

    car_control = AutonomousModelCarControl(vehicle_spec = vec_spec)

    rospy.init_node('drive_model_car', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
