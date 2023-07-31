import cv2
from ultralytics import YOLO
import torch
from helper import cmd_out, get_dir

class Tracker_with_pwm():
    def __init__(self, source = 0, weights = get_dir('yolov8n.pt'), tracker="bytetrack.yaml", show=True, stream=True):
        self.model = YOLO(weights)
        self.prev_id = 0
        self.run(source=source, show=show, stream=stream, tracker=tracker)

    def run(self, source, show, stream, tracker):
        for result in self.model.track(source=source, show=show, stream=stream, tracker=tracker):
            boxes = result.boxes
            current_id = boxes.id
            classes = boxes.cls

            # if there are detections and the detections contain objects of interest
            if current_id is not None and torch.any(classes == 0):
                tracked_index = torch.where(current_id == self.prev_id)[0]

                # if the tracked object is not there
                if tracked_index.numel() == 0:
                    # get the first item in the tensor of detected objects as the object to be tracked
                    self.prev_id = current_id[0].item()
                    print("\n\n\n\n")
                # if the tracked object is there
                else:
                    xywh = boxes.xywhn
                    center = xywh[tracked_index, 0:2].flatten()
                    box_dim = xywh[tracked_index, 2:4].flatten()
                    self.leftRightPwm, self.forwardBackwardPwm = cmd_out(center, box_dim, LeftRightThreshold=0.1, min_box_sz=0.16, max_box_sz=0.49)
                    # print("Left Right Pwm: ", self.leftRightPwm)
                    # print('Forward Backward Pwm: ', self.forwardBackwardPwm)

def main():
    tracker = Tracker_with_pwm(source=0)

if __name__=='__main__':
    main()

