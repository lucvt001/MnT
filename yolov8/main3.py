# import cv2
from ultralytics import YOLO
import torch
from helper import cmd_out, get_dir

# Load the YOLOv8 model
model = YOLO(get_dir('yolov8n.pt'))

def main():
    prev_id = 0
    for result in model.track(source=0, show=True, stream=True, tracker="bytetrack.yaml"):
        boxes = result.boxes
        current_id = boxes.id
        classes = boxes.cls

        # if there are detections and the detections contain objects of interest
        if current_id is not None and torch.any(classes == 0):
            tracked_index = torch.where(current_id == prev_id)[0]

            # if the tracked object is not there
            if tracked_index.numel() == 0:
                # get the first item in the tensor of detected objects as the object to be tracked
                prev_id = current_id[0].item()
                print("\n\n\n\n")
            # if the tracked object is there
            else:
                xywh = boxes.xywhn
                center = xywh[tracked_index, 0:2].flatten()
                print(center)
                box_dim = xywh[tracked_index, 2:4].flatten()
                leftRightPwm, forwardBackwardPwm = cmd_out(center, box_dim, LeftRightThreshold=0.1, min_box_sz=0.16, max_box_sz=0.49)
                print("Left Right Pwm: ", leftRightPwm)
                print('Forward Backward Pwm: ', forwardBackwardPwm)
        
if __name__=='__main__':
    main()

