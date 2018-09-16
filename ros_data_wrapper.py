import rospy
import rosbag
import cv2
from cv_bridge import CvBridge

class RosDataWrapper:

    def __init__(self, bagfiles = [], show_images = False):
        self.bagfiles = bagfiles
        self.bridge = CvBridge()

        self.show_images = show_images

    def read_ros_bag_file(self):

        for bagfile in self.bagfiles:
            bag = rosbag.Bag(bagfile,'r')

            for topic, msg, t in bag.read_messages():

                print(t, topic)

                if topic == '/camera/image_raw':
                    img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

                    if self.show_images:
                        cv2.imshow('test', img)
                        cv2.waitKey(1)

                if topic == '/CarUpdate':
                    print(topic)

            print(bagfile)



if __name__ == '__main__':

    files = ['/home/schuerlars/git/e2e_lane_keeping/velox_data/2013-01-01-01-47-52.bag']
    ros_data_handler = RosDataWrapper(files , show_images=True)

    ros_data_handler.read_ros_bag_file()