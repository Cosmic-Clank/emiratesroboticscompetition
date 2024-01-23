import math
from pupil_apriltags import Detector
import cv2
import numpy as np

class AprilTag(Detector):
    def __init__(self, 
                families="tagStandard41h12",
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0):
        
        super().__init__(families=families,
                        nthreads=nthreads,
                        quad_decimate=quad_decimate,
                        quad_sigma=quad_sigma,
                        refine_edges=refine_edges,
                        decode_sharpening=decode_sharpening,
                        debug=debug)
        
        self.results = []
    
    def detect(self, img):
        if img.shape != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        results = super().detect(img, estimate_tag_pose=True, camera_params=(1280, 720, 1280/2, 720/2), tag_size=0.05)
        if results:
            self.results = results
            # print(results)
        else:
            self.results = []
        
    def get_results(self):
        return self.results

    def get_drawn_image(self, img):
        marked_img = img.copy()

        for result in self.results:
            cv2.circle(marked_img, (int(result.center[0]), int(result.center[1])), 5, (0, 0, 255), -1)
            for i, point in enumerate(result.corners):
                cv2.circle(marked_img, (int(point[0]), int(point[1])), 8, (0, 255, 0), -1)
                cv2.putText(marked_img, str(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            direction_vector = result.corners[1] - result.corners[0]
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
            
            gripper_point = result.center + direction_vector * math.dist(result.corners[1], result.corners[0]) * 1.5
            result.gripper_point = gripper_point

            cv2.circle(marked_img, (int(gripper_point[0]), int(gripper_point[1])), 10, (255, 0, 0), -1)

        return marked_img