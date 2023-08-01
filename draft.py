from ultralytics import YOLO
import cv2

model = YOLO('yolov8/yolov8n-face.pt')
img = cv2.imread('/Users/tienlucvu/Programming/MnT/yolov8/download.jpeg')
res = model(img)