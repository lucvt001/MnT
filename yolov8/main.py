import cv2
from ultralytics import YOLO
import torch
from helper import *

# Load the YOLOv8 model
model = YOLO(get_dir('yolov8n-face.pt'))
device = get_dev()
model.to(device)

# Load camera
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        for result in model(frame, stream=True, device=device, show=True):
            boxes = result.boxes

            # Detect if there is any human inside the frame
            classes = boxes.cls
            xywh = boxes.xywhn
            if torch.any(classes == 0):      # 0 is the index for human class
                indices = (classes == 0).nonzero().flatten()         # check for the indices where the pred is human
                centers = xywh[indices][:, 0:2]
                print(centers)

            # Visualize the results on the frame
            # annotated_frame = result.plot()

            # Display the annotated frame
            # cv2.imshow("YOLOv8 Inference", annotated_frame)

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
