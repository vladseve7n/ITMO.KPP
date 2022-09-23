import os
import argparse
import shutil
import cv2
from car_detection import CarDetector
from plate_number_detection import PlateNumberDetector
from checkpoint_passing import is_checkpoint_passed, is_plate_number_allowed
from util.video import convert_images_to_video
from util.visualisation import draw_textlines, draw_checkpoint_line, draw_rectangles 


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

    car_detector = CarDetector()
    plate_number_detector = PlateNumberDetector()

    frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break 
        # 1. Detect cars on the frame
        car_bboxes = car_detector.detect(frame)
        # 2. Detect if car passed the checkpoint line
        # 3. crop car bounding box
        # 4. Detect plate number of the nearest car
        # Use some standard library like Nomeroff
        # 5. Open the gate if plate number in the list of allowed number plates.
        # Or close the gate if there are no allowed cars nearby
        # 6. Visualisation
        res_frame = frame.copy()
        draw_checkpoint_line(res_frame, CHECKPOINT_LINE_COORDS)
        draw_rectangles(res_frame, car_bboxes)
        draw_textlines(res_frame, ['Car passed', 'Checkpoint opened'])
        cv2.imwrite(f'{TEMP_FOLDER_PATH}/{frame_num}.jpg',res_frame)
        frame_num += 1
    cap.release()
    cv2.destroyAllWindows()
    convert_images_to_video(TEMP_FOLDER_PATH, f'{output_path}/result_video_from_camera_1.mp4')
    # Remove temp dir
    shutil.rmtree(TEMP_FOLDER_PATH)


if __name__ == '__main__':
    main()