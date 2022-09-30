import math
from itertools import groupby
from typing import List

from PIL import Image
import cv2
import numpy as np
import torch
import torchvision

from nomeroff_net import pipeline
from nomeroff_net.image_loaders import DumpyImageLoader
from nomeroff_net.tools import unzip
from ocr_model.model import OCR_CRNN


def reshape_points(target_points: List or np.ndarray, start_idx: int) -> List:
    if start_idx > 0:
        part1 = target_points[:start_idx]
        part2 = target_points[start_idx:]
        target_points = np.concatenate((part2, part1))
    return target_points


def build_perspective(img: np.ndarray, rect: list, w: int, h: int) -> List:
    img_h, img_w, img_c = img.shape
    if img_h < h:
        h = img_h
    if img_w < w:
        w = img_w
    pts1 = np.float32(rect)
    pts2 = np.float32(np.array([[0, 0], [w, 0], [w, h], [0, h]]))
    moment = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, moment, (w, h))


def find_distances(points: np.ndarray or List) -> List:
    def distance(p0: List or np.ndarray, p1: List or np.ndarray) -> float:
        """
        distance between two points p0 and p1
        """
        return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

    def fline(p0: List, p1: List) -> List:
        """
        Вычесление угла наклона прямой по 2 точкам
        """
        x1 = float(p0[0])
        y1 = float(p0[1])

        x2 = float(p1[0])
        y2 = float(p1[1])

        if x1 - x2 == 0:
            k = math.inf
            b = y2
        else:
            k = (y1 - y2) / (x1 - x2)
            b = y2 - k * x2
        r = math.atan(k)
        a = math.degrees(r)
        a180 = a
        if a < 0:
            a180 = 180 + a
        return [k, b, a, a180, r]

    def linear_line_matrix(p0: List, p1: List, verbode: bool = False) -> np.ndarray:
        """
        Вычесление коефициентов матрицы, описывающей линию по двум точкам
        """
        x1 = float(p0[0])
        y1 = float(p0[1])

        x2 = float(p1[0])
        y2 = float(p1[1])

        matrix_a = y1 - y2
        matrix_b = x2 - x1
        matrix_c = x2 * y1 - x1 * y2
        if verbode:
            print("Уравнение прямой, проходящей через эти точки:")
            print("%.4f*x + %.4fy = %.4f" % (matrix_a, matrix_b, matrix_c))
            print(matrix_a, matrix_b, matrix_c)
        return np.array([matrix_a, matrix_b, matrix_c])

    distanses = []
    cnt = len(points)

    for i in range(cnt):
        p0 = i
        if i < cnt - 1:
            p1 = i + 1
        else:
            p1 = 0
        distanses.append({"d": distance(points[p0], points[p1]), "p0": p0, "p1": p1,
                          "matrix": linear_line_matrix(points[p0], points[p1]),
                          "coef": fline(points[p0], points[p1])})
    return distanses


def get_cv_zone_rgb(img: np.ndarray, rect: list, gw: float = 0, gh: float = 0,
                    coef: float = 4.6, auto_width_height: bool = True) -> List:
    if gw == 0 or gh == 0:
        distanses = find_distances(rect)
        h = (distanses[0]['d'] + distanses[2]['d']) / 2
        if auto_width_height:
            w = int(h * coef)
        else:
            w = (distanses[1]['d'] + distanses[3]['d']) / 2
    else:
        w, h = gw, gh
    return build_perspective(img, rect, int(w), int(h))


def crop_number_plate_zones_from_images(images, images_points):
    zones = []
    for image, image_points in zip(images, images_points):
        image_zones = [get_cv_zone_rgb(image, reshape_points(rect, 1)) for rect in image_points]
        for zone in image_zones:
            zones.append(zone)
    return zones


def prepare_plate_for_ocr(frame, plate_polygon):
    plate_num_img = crop_number_plate_zones_from_images(frame, plate_polygon)
    # plate_num_img = np.array(plate_num_img[0], dtype='uint8')
    # cv2.imshow('plate', plate_num_img[0])
    # cv2.waitKey()
    return plate_num_img[0]


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
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                # torchvision.transforms.ToTensor(),
                                # torchvision.transforms.Resize((64, 256)),
                                # torchvision.transforms.Normalize(
                                #     mean=[0.5] * 3,
                                #     std=[0.5] * 3)
                            ])

    def detect(self, image: np.ndarray) -> np.array:
        images_bboxs, images = unzip(self.number_plate_localization([image]))
        images_points, _ = unzip(self.number_plate_key_points_detection(unzip([images, images_bboxs])))
        if len(images_points[0]) > 0:
            return np.array(images_points[0], dtype='int')[0].reshape(-1, 1, 2), images_points
        else:
            return None, None

    def detect_images(self, images: List[np.ndarray]) -> List[np.array]:
        images_bboxs, images = unzip(self.number_plate_localization(images))
        images_points, _ = unzip(self.number_plate_key_points_detection(unzip([images, images_bboxs])))
        all_points = []
        for points in images_points:
            all_points.append(np.array(points, dtype='int')[0].reshape(-1, 1, 2))
        return all_points

    def recognize_numplate_text(self, numplate_img) -> str:
        numplate_img = Image.fromarray(np.uint8(numplate_img))
        numplate_img = self.image_transform(numplate_img)
        numplate_img = torch.unsqueeze(numplate_img, 0)
        with torch.no_grad():
            predicted_tokens = self.ocr_model(numplate_img)
        _, max_index = torch.max(predicted_tokens, dim=2)
        raw_prediction = list(max_index[0].detach().cpu().numpy())
        prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_token]).cuda()
        numplate_str = ''.join([TOKEN2SYMBOLS_MAPPING[token] for token in prediction.tolist() if token != blank_token])
        return numplate_str
