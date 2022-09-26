import os
import argparse
import shutil
import cv2
from car_detection import CarDetector
from plate_number_detection import PlateNumberDetector
from checkpoint_passing import is_checkpoint_passed, is_plate_number_allowed
from util.video import convert_images_to_video
from util.visualisation import draw_textlines, draw_checkpoint_line, draw_rectangles, draw_plate_polygon


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
    cap = cv2.VideoCapture(video_paths[0]) # TODO: add support for multiple video/camera processing 
    if not cap.isOpened():
        print("Error opening video stream or file")
    TEMP_FOLDER_PATH = 'temp'
    os.makedirs(TEMP_FOLDER_PATH, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    CHECKPOINT_LINE_COORDS = [(150, 0), (150, 720)] # TODO: ask for these coords as cli args

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path + '/output.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    car_detector = CarDetector()
    plate_number_detector = PlateNumberDetector()

    frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break 
        # 1. Detect cars on the frame
        car_bboxes = car_detector.detect(frame)
        plate_polygon = plate_number_detector.detect(frame)
        # 2. Detect if car passed the checkpoint line
        # 3. crop car boundin  g box
        # 4. Detect plate number of the nearest car
        # Use some standard library like Nomeroff
        # 5. Open the gate if plate number in the list of allowed number plates.
        # Or close the gate if there are no allowed cars nearby
        # 6. Visualisation
        res_frame = frame.copy()
        draw_checkpoint_line(res_frame, CHECKPOINT_LINE_COORDS)
        draw_rectangles(res_frame, car_bboxes)
        draw_plate_polygon(res_frame, plate_polygon)

        draw_textlines(res_frame, ['Car passed', 'Checkpoint opened'])
        cv2.imshow('test', res_frame)
        cv2.waitKey(0)
        out.write(res_frame)

    cap.release()
    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    main()
