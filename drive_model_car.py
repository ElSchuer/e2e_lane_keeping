import numpy as np
import data_handler
import cnn_model
import scipy.misc
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class AutonomousModelCarControl:

    def __init__(self, vehicle_spec):
        self.vehicle_spec = vehicle_spec
        self.sub = rospy.Subscriber("chatter", Image, self.predict_steering_angle)
        self.bridge = CvBridge()

    def predict_steering_angle(self, image_msg):
        try:
            image = np.asarray(self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough'))
        except CvBridgeError as e:
            print(e)

        image = scipy.misc.imresize(image[self.vehicle_spec.image_crop_vert[0]:self.vehicle_spec.image_crop_vert[1]],[66, 200] / 255.0)-0.5

        #Calculate new Steering angle based on image input
        steering_angle = cnn_model.y.eval(feed_dict={cnn_model.x: image[None, :, :, :], cnn_model.keep_prob: 1.0})[0][0]

        return steering_angle*self.vehicle_spec.angle_norm


if __name__ == '__main__':
    vec_spec = data_handler.VehicleSpec(angle_norm=30, image_crop_vert=[220, 480])

    car_control = AutonomousModelCarControl(vehicle_spec = vec_spec)

    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
