import cv2
from ultralytics import YOLO
import torch

# Load the YOLOv8 model
model = YOLO('yolov8/yolov8n.pt')

# Load camera
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        result = model(frame)[0]
        boxes = result.boxes

        # Detect if there is any human inside the frame
        classes = boxes.cls
        xywh = boxes.xywhn
        if torch.any(classes == 0):      # 0 is the index for human class
            indices = (classes == 0).nonzero().flatten()         # check for the indices where the pred is human
            centers = xywh[indices][:, 0:2]
            print(centers)

        # Visualize the results on the frame
        annotated_frame = result.plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
