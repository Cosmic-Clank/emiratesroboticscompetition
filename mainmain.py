from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_perception_modules.armtag import InterbotixArmTagInterface
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface
import modules.object_detection as od
import modules.realsense as rs
import numpy as np
import cv2
import time
import pickle
import json

def get_labels(bounding_boxes):
    bounding_boxes = sorted(bounding_boxes, key=lambda bounding_box: bounding_box.center[0])
    labels = [bounding_box.class_name for bounding_box in bounding_boxes]
    print("labels: ", labels)
    return labels
    
def main():
    with open('bounding_boxes.pkl', 'rb') as f:
        bounding_boxes = pickle.load(f)
        object_labels = get_labels(bounding_boxes)
        
    with open("configuration.json", "r") as f:
        configuration = json.load(f)
        print("configuration: ", configuration)
        
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
    for index, cluster in enumerate(clusters):
        x, y, z = cluster["position"]
        bot.arm.set_ee_pose_components(x=x, y=y, z=z+0.05, pitch=0.5)
        bot.arm.set_ee_pose_components(x=x, y=y, z=z, pitch=0.5)
        bot.gripper.close()
        bot.arm.set_ee_pose_components(x=x, y=y, z=z+0.05, pitch=0.5)
        
        if "metal" in object_labels[index].lower():
            bot.arm.set_ee_pose_components(x=configuration["metal"]["x"], y=configuration["metal"]["y"], z=0.2)
            
        elif "plastic" in object_labels[index].lower():
            bot.arm.set_ee_pose_components(x=configuration["plastic"]["x"], y=configuration["plastic"]["y"], z=0.2)
            
        elif "paper" in object_labels[index].lower():
            bot.arm.set_ee_pose_components(x=configuration["paper"]["x"], y=configuration["paper"]["y"], z=0.2)
            
        else:
            bot.arm.set_ee_pose_components(x=configuration["glass"]["x"], y=configuration["glass"]["y"], z=0.2)
            
        bot.gripper.open()
        bot.arm.set_ee_pose_components(x=0.3, z=0.2)
    bot.arm.go_to_sleep_pose()

if __name__=='__main__':
    main()