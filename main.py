import argparse
import os
from typing import Union

import cv2
from pydantic import BaseModel

from car_detection import CarDetector
from plate_number_detection import PlateNumberDetector, prepare_plate_for_ocr
from util.visualisation import draw_info_table, draw_checkpoint_line, draw_rectangles, draw_plate_polygon, \
    draw_interested_area


class FrameInfo(BaseModel):
    car_colour: Union[str, None] = None
    car_type: Union[str, None] = None
    car_plate: Union[str, None] = None
    is_car_passed: bool = False


def main():
    parser = argparse.ArgumentParser(description='Process inputs.')
    parser.add_argument('--videos', type=str,
                        help='path to videos for inference', default='test_videos/test_video_cars.mp4')
    parser.add_argument('--allowed_cars_info', type=str,
                        help='path to file with info about allowed cars', default='test_allowed_cars_info.csv')
    parser.add_argument('--output_path', type=str,
                        help='path to output folder', default='output')
    args = parser.parse_args()
    video_paths = args.videos.split(',')
    output_path = args.output_path
    allowed_cars_info_path = args.allowed_cars_info

    # Open video
    cap = cv2.VideoCapture(video_paths[0])  # TODO: add support for multiple video/camera processing
    if not cap.isOpened():
        print("Error opening video stream or file")
    os.makedirs(output_path, exist_ok=True)

    CHECKPOINT_LINE_COORDS = [(150, 0), (150, 720)]  # TODO: ask for these coords as cli args
    INTERESTED_AREA = [(100, 0), (600, 720)]
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path + '/output.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width + 400, frame_height))

    car_detector = CarDetector()
    plate_number_detector = PlateNumberDetector(
        ocr_checkpoint_path='/home/dmitriy/projects/ITMO.KPP/ocr_model/lightning_logs/version_10/checkpoints/epoch=39-step=1199.ckpt')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 1. Detect cars on the frame
        print('.', end='')
        frame_info = FrameInfo()
        res_frame = frame.copy()
        frame = frame[INTERESTED_AREA[0][1]:INTERESTED_AREA[1][1], INTERESTED_AREA[0][0]:INTERESTED_AREA[1][0], :]

        car_bboxes = car_detector.detect(frame)
        if len(car_bboxes) == 0:
            res_frame = draw_info_table(res_frame, frame_info)
            draw_checkpoint_line(res_frame, CHECKPOINT_LINE_COORDS)
            draw_interested_area(res_frame, INTERESTED_AREA)
            out.write(res_frame)
            continue
        else:
            frame_info.car_colour = 'Undefined'
            frame_info.car_type = 'Undefined'
        # 2. Detect if car passed the checkpoint line
        # 3. crop car bounding box
        # 4. Detect plate number of the nearest car
        plate_polygon = plate_number_detector.detect(frame)
        num_plate_img = prepare_plate_for_ocr(frame, plate_polygon)
        if plate_polygon is not None:
            frame_info.car_plate = plate_number_detector.recognize_numplate_text(num_plate_img)
            #frame_info.car_plate = "Undefined"
        # 5. Open the gate if plate number in the list of allowed number plates.
        # Or close the gate if there are no allowed cars nearby

        draw_rectangles(frame, car_bboxes)
        draw_plate_polygon(frame, plate_polygon)

        res_frame[INTERESTED_AREA[0][1]:INTERESTED_AREA[1][1], INTERESTED_AREA[0][0]:INTERESTED_AREA[1][0], :] = frame

        draw_checkpoint_line(res_frame, CHECKPOINT_LINE_COORDS)
        draw_interested_area(res_frame, INTERESTED_AREA)
        res_frame = draw_info_table(res_frame, frame_info)
        out.write(res_frame)
        # cv2.imshow('test', res_frame)
        # cv2.waitKey(0)


    cap.release()
    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    main()
