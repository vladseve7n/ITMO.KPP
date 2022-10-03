import time
from typing import Union, List

import cv2
import numpy as np
from pydantic import BaseModel

from util.visualisation import draw_rectangles, draw_checkpoint_line, draw_interested_area, \
    draw_info_table


class FrameInfo(BaseModel):
    car_colour: Union[str, None] = None
    car_type: Union[str, None] = None
    car_plate: Union[str, None] = None
    is_car_passed: bool = False
    is_car_have_access: bool = False


class Camera:
    def __init__(self, camera: dict, output_path: str, visualize: bool = True, width: int = 800):
        self.visualize = visualize
        self.interested_area = camera['interested_area']
        self.cp_line = camera['check_point_line_coords']
        self.cap = cv2.VideoCapture(camera['camera_path'])
        self.name = camera['camera_name']
        self.number_plates = camera['number_plates']
        self.info = FrameInfo()
        self.work = self.cap.isOpened()
        self.last_open_request = 0
        self.last_car_passed = 0
        self.last_plates_dict = {}
        if not self.work:
            print(f"{self.name} doesn't work")
        else:
            if self.visualize:
                frame_width = int(self.cap.get(3))
                frame_height = int(self.cap.get(4))
                self.resize_coef = width / frame_width

                self.interested_area = np.array(np.array(self.interested_area) * self.resize_coef, dtype='int32')
                self.cp_line = np.array(np.array(self.cp_line) * self.resize_coef, dtype='int32')

                self.out = cv2.VideoWriter(output_path + f'/{self.name}.avi',
                                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                           (int(frame_width * self.resize_coef) + 400,
                                            int(frame_height * self.resize_coef)))
        self.image = None
        self.frame = None
        self.car_bboxes = []
        self.plate_polygon = None
        self.num_plate_img = None

    def get_image(self) -> np.array:
        if self.work:
            ret, frame = self.cap.read()
            if ret:
                dim = (int(frame.shape[1] * self.resize_coef), int(frame.shape[0] * self.resize_coef))
                frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                self.image = frame
                self.frame = frame[self.interested_area[0][1]:self.interested_area[1][1],
                             self.interested_area[0][0]:self.interested_area[1][0], :]

                return frame
            else:
                self.work = False
                self.image = None
                self.frame = None
        return None

    def set_car_bboxes(self, car_bboxes: List[np.array]):
        self.car_bboxes = car_bboxes

    def get_car_image(self):
        car_box = self.car_bboxes[0]  # TODO: find nearest car
        return self.frame[int(car_box[1]):int(car_box[3]), int(car_box[0]):int(car_box[2]), :]

    def detect_if_car_is_passed(self, num_of_points: int = 10):
        check_points = np.linspace(self.cp_line[0], self.cp_line[1], num_of_points, True)
        car_box = self.car_bboxes[0]  # TODO: find nearest car
        x_shift = self.interested_area[0][0]
        y_shift = self.interested_area[0][1]
        passed_points = 0
        for p in check_points:

            if_x_in_square = car_box[0] + x_shift < p[0] < car_box[2] + x_shift
            if_y_in_square = car_box[1] + y_shift < p[1] < car_box[3] + y_shift

            if if_x_in_square and if_y_in_square:
                passed_points += 1

        if passed_points > 0:
            self.info.is_car_passed = True
            if self.info.car_plate in self.last_plates_dict:
                if not (time.time() - self.last_plates_dict[self.info.car_plate] < 60):
                    self.last_plates_dict[self.info.car_plate] = time.time()
                    print(f'{self.name}: ({self.info}) is passed')
            else:
                self.last_plates_dict[self.info.car_plate] = time.time()
                print(f'{self.name}: ({self.info}) is passed')
        else:
            self.info.is_car_passed = False

    def set_plate(self, num_plate_img: np.array):
        self.num_plate_img = num_plate_img

    def check(self) -> bool:
        return self.work

    def preparation(self):
        if self.visualize:
            if len(self.car_bboxes) > 0:
                draw_rectangles(self.frame, self.car_bboxes)

            self.image[self.interested_area[0][1]:self.interested_area[1][1],
            self.interested_area[0][0]:self.interested_area[1][0], :] = self.frame

            if self.info.car_plate is not None:
                self.info.is_car_have_access = self.info.car_plate.upper() in self.number_plates

            if self.info.is_car_have_access and time.time() - self.last_open_request > 5:
                self.last_open_request = time.time()
                print(f'{self.name}: OPEN THE GATE REQUEST')

            draw_checkpoint_line(self.image, self.cp_line)
            draw_interested_area(self.image, self.interested_area)
            self.image = draw_info_table(self.image, self.info, num_plate=self.num_plate_img)
            self.out.write(self.image)

    def __del__(self):
        self.cap.release()
        if self.visualize:
            self.out.release()
        print(f"{self.name} is closed")


def check_cameras(cameras: List[Camera]) -> List[Camera]:
    check = [camera for camera in cameras if camera.check()]
    return check


def check_if_car_exists(car_bboxes: List[np.array], cameras: List[Camera]):
    cameras_with_cars = []
    for car_bbox, camera in zip(car_bboxes, cameras):
        if len(car_bbox) != 0:
            camera.set_car_bboxes(car_bbox)
            cameras_with_cars.append(camera)
        else:
            camera.info.car_type = None
            camera.info.car_colour = None
    return cameras_with_cars


def clean_cameras_where_car_is_passed(cameras: List[Camera]):
    not_passed_cars = []
    for camera in cameras:
        if not camera.info.is_car_passed:
            not_passed_cars.append(camera)

    return not_passed_cars
