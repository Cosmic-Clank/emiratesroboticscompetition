from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_perception_modules.armtag import InterbotixArmTagInterface
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface
import modules.object_detection as od
import modules.ros_image_listener as rir
import numpy as np
import cv2
import time
import json
import math
    
def main():
    object_detector = od.ObjectDetection()
    image_reader = rir.ROSImageReader("/camera/color/image_raw")
        
    with open("configuration.json", "r") as f:
        configuration = json.load(f)
        print("configuration: ", configuration)
        
    # Initialize the arm module along with the pointcloud and armtag modules
    bot = InterbotixManipulatorXS("wx250", moving_time=1.5, accel_time=0.75)
    pcl = InterbotixPointCloudInterface()
    armtag = InterbotixArmTagInterface()

    # set initial arm and gripper pose
    # bot.arm.set_ee_pose_components(x=configuration["april_tag_scan"]["x"], z=configuration["april_tag_scan"]["z"])
    bot.arm.go_to_sleep_pose()
    bot.gripper.open()

    # get the ArmTag pose
    bot.arm.set_ee_pose_components(y=-0.3, z=0.2)
    time.sleep(0.5)
    armtag.find_ref_to_arm_base_transform()
    bot.arm.set_ee_pose_components(x=configuration["april_tag_scan"]["x"], y=configuration["april_tag_scan"]["y"], z=configuration["april_tag_scan"]["z"])

    # get the cluster positions
    # sort them from max to min 'x' position w.r.t. the 'wx250/base_link' frame
    success, clusters = pcl.get_cluster_positions(ref_frame="wx250/base_link", sort_axis="x", reverse=True)

    # pick up all the objects and drop them in a virtual basket in front of the robot
    for cluster in clusters:
        x, y, z = cluster["position"]
        
        offset_x = -(x / (math.sqrt(x**2 + y**2))) * 0.05
        offset_y = -(y / (math.sqrt(x**2 + y**2))) * 0.05
        
        bot.arm.set_ee_pose_components(x=x+offset_x, y=y+offset_y, z=z+0.1)
        bot.arm.set_ee_pose_components(x=x+offset_x, y=y+offset_y, z=z-0.05)
        bot.gripper.close()
        bot.arm.set_ee_pose_components(x=0, y=0, z=0.2)
        
        bot.arm.set_ee_pose_components(x=configuration["april_tag_scan"]["x"], y=configuration["april_tag_scan"]["y"], z=configuration["april_tag_scan"]["z"])
        
        material = object_detector.infer_image_from_cv2(image_reader.get_latest_image())
        print("Material Identified: ", material)
        
        bot.arm.set_ee_pose_components(x=0.02, y=0.3, z=0.2)
        
        bot.arm.set_single_joint_position("waist", np.pi/2.0)
        
        if "metal" in material:
            bot.arm.set_ee_pose_components(x=configuration["metal"]["x"], y=configuration["metal"]["y"], z=configuration["metal"]["z"])
            
        elif "plastic" in material:
            bot.arm.set_ee_pose_components(x=configuration["plastic"]["x"], y=configuration["plastic"]["y"], z=configuration["plastic"]["z"])
            
        elif "paper" in material:
            bot.arm.set_ee_pose_components(x=configuration["paper"]["x"], y=configuration["paper"]["y"], z=configuration["paper"]["z"])
            
        else:
            bot.arm.set_ee_pose_components(x=configuration["glass"]["x"], y=configuration["glass"]["y"], z=configuration["glass"]["z"])
            
        bot.gripper.open()
        bot.arm.set_ee_pose_components(x=0, y=0, z=0.2)

    bot.arm.go_to_sleep_pose()

if __name__=='__main__':
    main()