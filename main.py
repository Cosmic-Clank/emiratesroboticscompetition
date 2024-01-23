from interbotix_xs_modules.arm import InterbotixManipulatorXS

from modules.apriltag import AprilTag
from modules.realsense import RealSense
from modules.object_detection import ObjectDetection

import cv2
import math
import threading

class CameraViewer(threading.Thread):
    def __init__(self, camera, object_detector, april_tag_detector):
        super().__init__()
        self.camera = camera
        self.object_detector = object_detector
        self.april_tag_detector = april_tag_detector
        
    def run(self):
        while True:
            color_image, _, _ = self.camera.capture_frame()
            
            self.object_detector.infer_image(color_image)
            drawn_image = self.object_detector.get_infered_image()
            
            self.april_tag_detector.detect(color_image)
            drawn_image = self.april_tag_detector.get_drawn_image(drawn_image)
            
            cv2.imshow('RealSense', drawn_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

def main():
    default_yolo_model = ("yolo-Weights/yolov8x.pt", ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"])
    
    at_detector = AprilTag()
    camera = RealSense()
    object_detector = ObjectDetection(*default_yolo_model)
    
    camera_viewer = CameraViewer(camera, object_detector, at_detector)
    camera_viewer.start()
    
    bot = InterbotixManipulatorXS("wx250", moving_time=1.5, accel_time=0.75)
    
    bot.arm.go_to_home_pose()
    bot.gripper.open()
    bot.arm.set_ee_pose_components(x=0.3, z=0.1)
    
    at_detector.detect(camera.capture_frame()[0])
    if not at_detector.get_results():
        raise Exception("No AprilTag detected")
    initial_position_px = at_detector.get_results()[0].center
    
    offset = 0.2
    
    bot.arm.set_ee_pose_components(x=0.3+offset, z=0.1)
    
    at_detector.detect(camera.capture_frame()[0])
    if not at_detector.get_results():
        raise Exception("No AprilTag detected")
    final_position_px = at_detector.get_results()[0].center
    
    print("Initial position: ", initial_position_px, "Final position: ", final_position_px)
    
    px_per_m_ratio = math.dist(initial_position_px, final_position_px) / offset
    
    print("Pixels per meter ratio: ", px_per_m_ratio)
    
    bot.arm.go_to_home_pose()
    at_detector.detect(camera.capture_frame()[0])
    if not at_detector.get_results():
        raise Exception("No AprilTag detected")
    
    bot_origin_px = (at_detector.get_results()[0].center[0] - (0.433*px_per_m_ratio), at_detector.get_results()[0].center[1])
    
    bot.arm.go_to_sleep_pose()
    
    # Detect objects and do pick and drop
    color_image, _, _ = camera.capture_frame()
    object_detector.infer_image(color_image)
        
    od_results = object_detector.get_bounding_boxes()
    for result in od_results:
        x = result.center[0] - bot_origin_px[0]
        y = result.center[1] - bot_origin_px[1]
        y*=-1
        print("Object at: ", x, y)
        bot.arm.set_ee_pose_components(x=x/px_per_m_ratio, y=y/px_per_m_ratio, z=0.1)
        bot.arm.set_ee_cartesian_trajectory(z=0.05)
        bot.gripper.close()
        bot.arm.set_ee_cartesian_trajectory(z=0.1)
        bot.gripper.open()
        
        bot.arm.go_to_sleep_pose()