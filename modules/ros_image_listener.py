import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ROSImageReader:
    def __init__(self, topic_name):
        self.latest_image = None
        self.bridge = CvBridge()
        self.topic_name = topic_name
        rospy.init_node('image_listener', anonymous=True)
        rospy.Subscriber(self.topic_name, Image, self.image_callback)

    def image_callback(self, data):
        # Convert ROS Image message to OpenCV image
        self.latest_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def get_latest_image(self):
        return self.latest_image

if __name__ == '__main__':
    # Initialize the ROS node and create an instance of ROSImageReader
    reader = ROSImageReader("/camera/color/image_raw")

    # Read images on-demand
    while not rospy.is_shutdown():
        # Perform other tasks here
        # For demonstration purposes, we're displaying the latest image
        cv2.imshow("Latest Image", reader.get_latest_image())
        
        if cv2.waitKey(0) == ord("q"):
            break
