from ultralytics import YOLO
import pickle
import cv2
import math

class ObjectDetection:
    def __init__(self, path_to_model, labels):
        self.model = YOLO(path_to_model)
        self.labels = labels
        self.current_frame = None
        self.bounding_boxes = []
        
    def infer_image(self, np_image, depth_frame=None):
        results = self.model.predict(np_image, verbose=False, stream=True)
        self.current_frame = np_image
        self.bounding_boxes.clear()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100

                # class name
                class_name = int(box.cls[0])

                center = (int(x1 + (x2-x1)/2), int(y1 + (y2-y1)/2))
                
                if depth_frame:
                    distance = depth_frame.get_distance(center[0], center[1])
                else:
                    distance = 0
                
                self.bounding_boxes.append(_BoundingBox(x1, y1, x2, y2, confidence, self.labels[class_name], distance))
        
        # self.bounding_boxes = pickle.load(open("bounding_boxes.pkl", "rb"))
        
    def get_bounding_boxes(self):
        return self.bounding_boxes
        
    def get_infered_image(self, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, color=(255, 0, 0), font_thickness=2):
        infered_image = self.current_frame.copy()
                
        for bounding_box in self.bounding_boxes:
            cv2.rectangle(infered_image, (bounding_box.x1, bounding_box.y1), (bounding_box.x2, bounding_box.y2), (255, 255, 255), 3)
            cv2.circle(infered_image, bounding_box.center, 5, (0, 0, 255), -1)
            cv2.rectangle(infered_image, (bounding_box.x1, bounding_box.y1 - 20), (bounding_box.x2, bounding_box.y1), (255, 0, 255), -1)
            cv2.putText(infered_image, f"{bounding_box.class_name.upper()} CONF: {bounding_box.confidence} CORDS: {bounding_box.center} DIST: {bounding_box.distance:.2f}m", (bounding_box.x1, bounding_box.y1 - 5), font, font_scale, color, font_thickness)

        return infered_image
    
class _BoundingBox:
    def __init__(self, x1, y1, x2, y2, confidence, class_name, distance):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        
        self.width = x2 - x1
        self.height = y2 - y1
        
        self.confidence = confidence
        self.class_name = class_name
        self.center = self.find_center()

        self.distance = distance
        
    def find_center(self):
        return (int(self.x1 + (self.x2-self.x1)/2), int(self.y1 + (self.y2-self.y1)/2))