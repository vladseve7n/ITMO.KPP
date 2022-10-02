import torch


class CarDetector:
    def __init__(self) -> None:
        self.detector_model = torch.hub.load('ultralytics/yolov5',
                                             'yolov5m', pretrained=True)

    def detect(self, images):
        results = self.detector_model(images)
        car_bboxes = []
        for detected_bboxes in results.pandas().xyxy:
            car_bboxes.append(detected_bboxes[detected_bboxes['name'] == 'car'].iloc[:, :4].values)
        return car_bboxes

    def detect_car_colour(self, images):
        colours = []
        for img in images:
            colours.append('Undefined')
        return colours

    def detect_car_type(self, images):
        car_types = []
        for img in images:
            car_types.append('Undefined')
        return car_types
