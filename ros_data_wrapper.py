import rospy
import rosbag
import cv2
import os
import csv
from cv_bridge import CvBridge
from messages.msg import CarControlMessage
from std_msgs.msg import Float32
import scipy.misc
import glob

class RosDataWrapper:

    def __init__(self, input_path, output_path ,show_images = False):
        self.folder = input_path
        self.output_path = output_path

        self.bridge = CvBridge()


        self.show_images = show_images

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if not os.path.exists(self.output_path + 'IMG'):
            os.makedirs(self.output_path + 'IMG')

    def save_data(self, image, speed, angle, img_name):
        with open(self.output_path + 'data_labels.csv', mode='a+') as new_log_file:
            writer = csv.writer(new_log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([img_name, '', '', str(angle), str(speed)])

        scipy.misc.imsave(output_path + img_name, image)

    def read_ros_bag_file(self):
        bagfiles = glob.glob(self.folder + '/*.bag')
        bagfiles.extend(glob.glob(self.folder + '/**/*.bag'))
        bagfiles.extend(glob.glob(self.folder + '/**/**/*.bag'))

        img_count = 0

        for bagfile in bagfiles:
            bag = rosbag.Bag(bagfile,'r')

            is_new_img = False
            is_new_angle = False
            is_new_speed = False

            for topic, msg, t in bag.read_messages():

                if topic == '/camera/image_raw':
                    img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                    is_new_img = True

                    if self.show_images:
                        cv2.imshow('image', img)
                        cv2.waitKey(1)

                if topic == '/CarUpdate':
                    angle = msg.steeringAngle
                    speed = msg.speed
                    is_new_angle = True
                    is_new_speed = True

                if topic == '/ECU/SteeringAngle' or topic == '/vehicle_info/steering_angle':
                    angle = msg.data
                    is_new_angle = True

                if topic == '/ECU/Speed' or topic == '/vehicle_info/speed':
                    speed = msg.data
                    is_new_speed = True

                if is_new_angle and is_new_img and is_new_speed:
                    self.save_data(image=img, speed=speed, angle=angle, img_name='IMG/'+'image_'+str(img_count)+'.jpg')
                    is_new_angle = False
                    is_new_img = False
                    is_new_speed = False
                    img_count = img_count + 1

            print("Image count : ",img_count)
            print(bagfile)



if __name__ == '__main__':

    folder = '/home/elschuer/data/LaneKeepingE2E/data_train/'
    output_path = '/home/elschuer/data/LaneKeepingE2E/images_train/'

    ros_data_handler = RosDataWrapper(input_path=folder , output_path=output_path, show_images=True)

    ros_data_handler.read_ros_bag_file()
