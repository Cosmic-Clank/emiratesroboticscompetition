import time
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_perception_modules.armtag import InterbotixArmTagInterface
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface

import modules.object_detection as od
import modules.realsense as rs

import numpy as np
import cv2

# This script uses a color/depth camera to get the arm to find objects and pick them up.
# For this demo, the arm is placed to the left of the camera facing outward. When the
# end-effector is located at x=0, y=-0.3, z=0.2 w.r.t. the 'wx200/base_link' frame, the AR
# tag should be clearly visible to the camera. A small basket should also be placed in front of the arm.
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_perception xsarm_perception.launch robot_model:=wx200'
# Then change to this directory and type 'python pick_place.py'
def initial_setup():
    yolo_model = ("yolo-Weights/best.pt", [
    "Glass",
    "Metal",
    "PLASTIC",
    "Paper",
    "Paper-cups",
    "Plastic bag",
    "Plastic bottle",
    "Plastic can",
    "Plastic canister",
    "Plastic caps",
    "Plastic cup",
    "Plastic shaker",
    "Plastic shavings",
    "Plastic toys",
    "Plastic zip bag"
])
    
    object_detector = od.ObjectDetection(*yolo_model)
    camera = rs.RealSense()
    
    img_count = 1
    images = []
    for i in range(img_count):
        color_image, depth_image, depth_frame = camera.capture_frame()
        if not np.any(color_image) or not np.any(depth_image) or not np.any(depth_frame):
            continue
        images.append(color_image)
    camera.stop_camera()
        
    object_detector.infer_image(images[0], depth_frame)
    
    bounding_boxes = object_detector.get_bounding_boxes()
    cv2.imshow('RealSense', object_detector.get_infered_image())
    return sorted(bounding_boxes, key=lambda bounding_box: bounding_box.center[0])
    
def main():
    # Initialize the arm module along with the pointcloud and armtag modules
    bot = InterbotixManipulatorXS("wx250", moving_time=1.5, accel_time=0.75)
    pcl = InterbotixPointCloudInterface()
    # armtag = InterbotixArmTagInterface()

    # set initial arm and gripper pose
    bot.arm.set_ee_pose_components(x=0.3, z=0.2)
    bot.gripper.open()

    # get the ArmTag pose
    # bot.arm.set_ee_pose_components(y=-0.3, z=0.2)
    # time.sleep(0.5)
    # armtag.find_ref_to_arm_base_transform()
    # bot.arm.set_ee_pose_components(x=0.3, z=0.2)

    # get the cluster positions
    # sort them from max to min 'x' position w.r.t. the 'wx250/base_link' frame
    success, clusters = pcl.get_cluster_positions(ref_frame="wx250/base_link", sort_axis="x", reverse=True)

    # pick up all the objects and drop them in a virtual basket in front of the robot
    for cluster in clusters:
        x, y, z = cluster["position"]
        bot.arm.set_ee_pose_components(x=x, y=y, z=z+0.05, pitch=0.5)
        bot.arm.set_ee_pose_components(x=x, y=y, z=z, pitch=0.5)
        bot.gripper.close()
        bot.arm.set_ee_pose_components(x=x, y=y, z=z+0.05, pitch=0.5)
        bot.arm.set_ee_pose_components(x=0.3, z=0.2)
        bot.gripper.open()
    bot.arm.go_to_sleep_pose()

if __name__=='__main__':
    main()