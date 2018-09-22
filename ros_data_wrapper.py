import rospy
import rosbag
import cv2
import os
import csv
from cv_bridge import CvBridge
from messages.msg import CarControlMessage
import scipy.misc

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
        with open(self.output_path + 'augmented_log.csv', mode='a+') as new_log_file:
            writer = csv.writer(new_log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([img_name, '', '', str(angle), str(speed)])

        scipy.misc.imsave(output_path + img_name, image)

    def read_ros_bag_file(self):
        for bagfile in os.listdir(self.folder):
            bag = rosbag.Bag(self.folder + bagfile,'r')

            img_count = 0
            is_new_img = False
            is_new_angle = False

            for topic, msg, t in bag.read_messages():

                print(t, topic)

                if topic == '/camera/image_raw':
                    img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                    is_new_img = True

                    if self.show_images:
                        cv2.imshow('test', img)
                        cv2.waitKey(1)

                if topic == '/CarUpdate':
                    angle = msg.steeringAngle
                    speed = msg.speed
                    is_new_angle = True

                if is_new_angle and is_new_img:
                    self.save_data(image=img, speed=speed, angle=angle, img_name='IMG/'+bagfile[:len(bagfile)-4]+'_img_'+str(img_count)+'.jpg')
                    is_new_angle = False
                    is_new_img = False
                    img_count =img_count + 1

            print(bagfile)



if __name__ == '__main__':

    folder = '/home/schuerlars/git/e2e_lane_keeping/velox_data/'
    output_path = '/home/schuerlars/git/e2e_lane_keeping/velox_data_path/'
    ros_data_handler = RosDataWrapper(input_path=folder , output_path=output_path, show_images=True)

    ros_data_handler.read_ros_bag_file()