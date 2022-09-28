from curses import raw
from typing import List
from itertools import groupby

import cv2
import numpy as np
import torch
import torchvision
from nomeroff_net import pipeline
from nomeroff_net.image_loaders import DumpyImageLoader
from nomeroff_net.tools import unzip

from ocr_model.model import OCR_CRNN


def prepare_plate_for_ocr(frame, plate_polygon):
    plate_num_img = cv2.imread('A001BP54.png')
    plate_num_img = cv2.cvtColor(plate_num_img, cv2.COLOR_BGR2RGB) 
    return plate_num_img
    

TOKEN2SYMBOLS_MAPPING = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
        10: 'A', 11: 'B', 12: 'E', 13: 'K', 14: 'M', 15: 'H', 16: 'O', 17: 'P',
         18: 'C', 19: 'T', 20: 'Y', 21: 'X'
    }
blank_token = 22

class PlateNumberDetector:
    def __init__(self, ocr_checkpoint_path) -> None:
        dumpy_loader = DumpyImageLoader
        self.number_plate_key_points_detection = pipeline("number_plate_key_points_detection",
                                                          image_loader=dumpy_loader)

        self.number_plate_localization = pipeline("number_plate_localization",
                                                  image_loader=dumpy_loader)
        self.ocr_model = OCR_CRNN.load_from_checkpoint(ocr_checkpoint_path,
            data_dir='autoriaNumberplateOcrRu-2021-09-01/test/img')
        self.ocr_model.eval()
        self.image_transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Resize((64, 256)),
                                torchvision.transforms.Normalize(
                                    mean=[0.5] * 3,
                                    std=[0.5] * 3)
                            ])

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

    def recognize_numplate_text(self, numplate_img) -> str:
        numplate_img = self.image_transform(numplate_img)
        with torch.no_grad():
            predicted_tokens = self.ocr_model(numplate_img)
        _, max_index = torch.max(predicted_tokens, dim=2)
        raw_prediction = list(max_index[:, 0].detach().cpu().numpy())
        prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_token]).cuda()
        numplate_str = ''.join([TOKEN2SYMBOLS_MAPPING[token] for token in prediction.tolist() if token != blank_token])
        return numplate_str

