import cv2
import modules.object_detection as od
import pickle
import modules.realsense as rs

# Create an empty dictionary to store bounding box information
bounding_boxes = []

# Callback function for mouse events
def draw_rectangle(event, x, y, flags, param):
    global drawing, top_left_pt, bottom_right_pt

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)

        # Store bounding box information in the dictionary
        
        key = len(bounding_boxes) + 1
        bounding_boxes.append(od._BoundingBox(top_left_pt[0], top_left_pt[1], bottom_right_pt[0], bottom_right_pt[1], 1, "TRASH", 0))

        # Draw the bounding box on the image
        cv2.rectangle(img_copy, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
        cv2.putText(img_copy, str(key), (top_left_pt[0], top_left_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Create a named window and set the callback function
cv2.namedWindow("Snapshot")
cv2.setMouseCallback("Snapshot", draw_rectangle)

# Open the camera
camera = rs.RealSense()

# Capture a snapshot from the camera
img, _, _ = camera.capture_frame()
camera.stop_camera()

# Make a copy of the image for drawing
img_copy = img.copy()

# Variables for drawing rectangles
drawing = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)

while True:
    # Display the snapshot image
    cv2.imshow("Snapshot", img_copy)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cv2.destroyAllWindows()

pickle.dump(bounding_boxes, open("bounding_boxes.pkl", "wb"))