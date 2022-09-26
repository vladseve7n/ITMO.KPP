from typing import List

import numpy as np

from nomeroff_net import pipeline
from nomeroff_net.image_loaders import DumpyImageLoader
from nomeroff_net.tools import unzip


class PlateNumberDetector:
    def __init__(self) -> None:
        dumpy_loader = DumpyImageLoader
        self.number_plate_key_points_detection = pipeline("number_plate_key_points_detection",
                                                          image_loader=dumpy_loader)

        self.number_plate_localization = pipeline("number_plate_localization",
                                                  image_loader=dumpy_loader)

    def detect(self, image: np.ndarray) -> np.array:
        images_bboxs, images = unzip(self.number_plate_localization([image]))
        images_points, _ = unzip(self.number_plate_key_points_detection(unzip([images, images_bboxs])))
        if len(images_points[0]) > 0:
            return np.array(images_points[0], dtype='int')[0].reshape(-1, 1, 2)
        else:
            return None

    def detect_images(self, images: List[np.ndarray]) -> List[np.array]:
        images_bboxs, images = unzip(self.number_plate_localization(images))
        images_points, _ = unzip(self.number_plate_key_points_detection(unzip([images, images_bboxs])))
        all_points = []
        for points in images_points:
            all_points.append(np.array(points, dtype='int')[0].reshape(-1, 1, 2))
        return all_points
