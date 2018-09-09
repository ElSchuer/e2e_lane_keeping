import rospy
import rosbag
import cv2
from cv_bridge import CvBridge

class RosDataWrapper:

    def __init__(self, bagfiles = []):
        self.bagfiles = bagfiles
        self.bridge = CvBridge()

    def readRosBagFile(self):

        for bagfile in self.bagfiles:
            bag = rosbag.Bag(bagfile,'r')

            for topic, msg, t in bag.read_messages():

                print(t, topic)

                if topic == '/camera/image_raw':
                    img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

                    cv2.imshow('test', img)
                    cv2.waitKey(1)

            print(bagfile)



if __name__ == '__main__':

    files = ['/home/schuerlars/data/ros_data/Testfahrt 18_12_2017/2017-12-18-16-37-54.bag']
    ros_data_handler = RosDataWrapper(files)

    ros_data_handler.readRosBagFile()