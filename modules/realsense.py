import pyrealsense2 as rs
import numpy as np
import cv2

class RealSense:
    def __init__(self, capture_shape = (1280, 720), capture_framerate = 30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print("Initialising realsense camera")
        print("Camera product line:", device_product_line)

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
        
        self.shape = capture_shape
        self.framerate = capture_framerate
        
        print("Capturing resolution and framerate:")
        print("Resolution:", self.shape)
        print("Fps:", self.framerate)

        self.config.enable_stream(rs.stream.depth, self.shape[0], self.shape[1], rs.format.z16, self.framerate)
        self.config.enable_stream(rs.stream.color, self.shape[0], self.shape[1], rs.format.bgr8, self.framerate)

        # Start streaming
        self.pipeline.start(self.config)
        
    def capture_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return np.empty(0, 0), np.empty(0, 0), np.empty(0, 0)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        return color_image, depth_colormap, depth_frame
    
    def stop_camera(self):
        self.pipeline.stop()