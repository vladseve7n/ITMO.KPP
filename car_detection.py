import imp
import torch
import pandas as pd


class CarDetector:
    def __init__(self) -> None:
        self.detector_model = torch.hub.load('ultralytics/yolov5', 
            'yolov5m', pretrained=True)

    def detect(self, image):
        results = self.detector_model([image])
        car_bboxes = []
        detected_bboxes = results.pandas().xyxy[0]
        car_bboxes = detected_bboxes[detected_bboxes['name']=='car'].iloc[:,:4].values
        return car_bboxes