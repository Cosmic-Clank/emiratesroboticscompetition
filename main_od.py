import modules.realsense as rs
import modules.object_detection as od
import numpy as np
import cv2
import time

default_yolo_model = ("yolo-Weights/yolov8x.pt", ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"])

trained_trash_model = ("yolo-Weights/best.pt", [
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

object_detector = od.ObjectDetection(*default_yolo_model)

camera = rs.RealSense((1280, 720), 30)

try:
    frames_count = 0
    start_time = time.time()
    while True:
        color_image, depth_image, depth_frame = camera.capture_frame()
        if not np.any(color_image) or not np.any(depth_image) or not np.any(depth_frame):
            continue

        object_detector.infer_image(color_image, depth_frame)
        infered_image = object_detector.get_infered_image()

        cv2.imshow('RealSense', infered_image)

        frames_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

finally:
    print(f"Average FPS:    {frames_count / (time.time() - start_time)}")
    camera.stop_camera()
