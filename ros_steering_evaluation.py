import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt

class LaneKeepingValidator:

    def __init__(self, plot_error_values = True, show_image = True):
        self.gt_angle = 0
        self.pred_angle = 0
        self.error_values = []
        self.pos_errors = []
        self.neg_errors = []

        self.plot_error_values = plot_error_values
        self.show_image = show_image

        self.gt_angle_sub = rospy.Subscriber("/ECU/SteeringAngle", Float32, self.gt_angle_callback)
        self.pred_angle_sub = rospy.Subscriber("/control/steering_angle_predicted", Float32, self.pred_angle_callback)
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.get_image)
        self.bridge = CvBridge()

        plt.ion()
        plt.show()

    def gt_angle_callback(self, angle_msg):
        self.gt_angle = angle_msg.data

    def pred_angle_callback(self, angle_msg):
        self.pred_angle = angle_msg.data

    def get_image(self, image_msg):
        try:
            image = np.asarray(self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough'))

            if self.show_image:
                cv2.imshow('Image', image)
                cv2.waitKey(1)

            self.validate()

        except CvBridgeError as e:
            print(e)



    def validate(self):
        print('-------------------')
        print("Predicted Angle : " + str(self.pred_angle))
        print("Ground Truth Angle : " + str(self.gt_angle))

        error = np.sqrt(np.power(self.pred_angle-self.gt_angle, 2))

        if np.sign(self.gt_angle) < 0:
            self.neg_errors.append(error)
        elif np.sign(self.gt_angle) > 0:
            self.pos_errors.append(error)

        self.error_values.append(error)
        print("Error Value : " + str(error))
        print("Mean Error : " + str(np.mean(self.error_values)))
        print("Mean Neg Error : " + str(np.mean(self.neg_errors)))
        print("Mean Pos Error : " + str(np.mean(self.pos_errors)))

        # Plotting the error values
        if self.plot_error_values:
            x_values = np.arange(0 , len(self.error_values))
            plt.plot(x_values, self.error_values, 'C1')
            plt.xlabel('Sample Number')
            plt.ylabel('Steering Angle MSE')
            plt.draw()
            plt.pause(0000.1)




if __name__ == '__main__':

    car_control = LaneKeepingValidator(show_image=True, plot_error_values=True)

    rospy.init_node('steering_eval', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")



