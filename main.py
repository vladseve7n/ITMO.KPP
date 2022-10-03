import argparse
import gc
import json
import os

from cameras import Camera, check_cameras, check_if_car_exists, clean_cameras_where_car_is_passed
from car_detection import CarDetector
from plate_number_detection import PlateNumberDetector, prepare_plate_for_ocr


def main():
    parser = argparse.ArgumentParser(description='Process inputs.')
    parser.add_argument('--cameras', type=str,
                        help='path to info about cameras', default='cameras.json')
    parser.add_argument('--output_path', type=str,
                        help='path to output folder', default='output')
    parser.add_argument('--visualize', type=bool,
                        help='Is it necessary to visualize', default=True)

    args = parser.parse_args()
    cameras_paths = args.cameras
    output_path = args.output_path
    visualize = args.visualize
    os.makedirs(output_path, exist_ok=True)

    with open(cameras_paths, 'r') as source:
        dict_with_cameras = json.load(source)

    cameras = [Camera(camera, output_path, visualize) for camera in dict_with_cameras['cameras']]

    car_detector = CarDetector()
    plate_number_detector = PlateNumberDetector(ocr_checkpoint_path='misc/test_checkpoint.ckpt')

    print(gc.collect())

    while True:

        for camera in cameras:
            camera.get_image()

        cameras = check_cameras(cameras)

        if len(cameras) == 0:
            break

        car_bboxes = car_detector.detect([camera.frame for camera in cameras])

        cameras_with_cars = check_if_car_exists(car_bboxes, cameras)

        for camera in cameras_with_cars:
            camera.detect_if_car_is_passed()

        cars_that_do_not_passed = clean_cameras_where_car_is_passed(cameras_with_cars)

        car_images = [camera.get_car_image() for camera in cars_that_do_not_passed]

        car_colours = car_detector.detect_car_colour(car_images)
        car_types = car_detector.detect_car_type(car_images)

        for camera, colour, car_type in zip(cars_that_do_not_passed, car_colours, car_types):
            camera.info.car_type = car_type
            camera.info.car_colour = colour

        for camera, car_img in zip(cars_that_do_not_passed, car_images):
            plate_polygon, images_points = plate_number_detector.detect(car_img)
            num_plate_img = None
            if plate_polygon is not None:
                num_plate_img = prepare_plate_for_ocr([car_img], images_points)
                camera.info.car_plate = plate_number_detector.recognize_numplate_text(num_plate_img)
            else:
                camera.info.car_plate = None
            camera.set_plate(num_plate_img)

        for camera in cameras:
            camera.preparation()


if __name__ == '__main__':
    main()
