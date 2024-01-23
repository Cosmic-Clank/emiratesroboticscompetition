import modules.realsense as rs
import modules.apriltag as at
import cv2

camera = rs.RealSense()
at_detector = at.AprilTag()

while True:
    color_image, _, _ = camera.capture_frame()

    at_detector.detect(color_image)
    drawn_image = at_detector.get_drawn_image(color_image)
    at_detector.get_results()[0].gripper_point
    
    cv2.imshow('RealSense', drawn_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break