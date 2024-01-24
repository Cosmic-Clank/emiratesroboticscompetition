import pickle
import modules.realsense as rs
import modules.object_detection as od

import cv2
# Take a snapshot

default_yolo_model = ("yolo-Weights/yolov8x.pt", ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"])

camera = rs.RealSense()
object_detector = od.ObjectDetection(*default_yolo_model)

image, _, _ = camera.capture_frame()

# Perform object detection
object_detector.infer_image(image)

bounding_boxes = object_detector.get_bounding_boxes()
infered_image = object_detector.get_infered_image()

# Save the bounding boxes as a .pkl file
with open('bounding_boxes.pkl', 'wb') as f:
    pickle.dump(bounding_boxes, f)

# Save the inferred image
cv2.imwrite('infered_image.jpg', infered_image)

camera.stop_camera()