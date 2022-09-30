import cv2
import numpy as np


def scale_image(img, scale_factor: int = 900):
    scaler = max(img.shape[0], img.shape[1]) / scale_factor
    if scaler == 0:
        return img
    return cv2.resize(img, (int(img.shape[1] / scaler),
                            int(img.shape[0] / scaler)), cv2.INTER_NEAREST)


def draw_info_table(frame, frame_info, num_plate) -> np.array:
    info_table = np.zeros((frame.shape[0], 400, 3), dtype='uint8')
    cv2.putText(img=info_table, text=f'Car colour: {frame_info.car_colour}', org=(5, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)

    cv2.putText(img=info_table, text=f'Car type: {frame_info.car_type}', org=(5, 80),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)

    cv2.putText(img=info_table, text=f'Car plate: {frame_info.car_plate}', org=(5, 110),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)

    if num_plate is not None:
        num_plate = scale_image(num_plate, 200)
        info_table[150:150+num_plate.shape[0], :num_plate.shape[1]] = num_plate

    frame = np.hstack((frame, info_table))
    return frame


def draw_interested_area(frame, INTERESTED_AREA):
    cv2.rectangle(frame, (INTERESTED_AREA[0][0], INTERESTED_AREA[0][1]),
                  (INTERESTED_AREA[1][0], INTERESTED_AREA[1][1]), (200, 120, 50), 2)


def draw_checkpoint_line(frame, line_coords):
    cv2.line(frame, line_coords[0], line_coords[1], (255, 0, 0), 5)


def draw_rectangles(frame, rectangles):
    for rectangle in rectangles:
        # print((round(rectangle[0]), round(rectangle[1])),
        #    (round(rectangle[2]), round(rectangle[3])))
        cv2.rectangle(frame, (round(rectangle[0]), round(rectangle[1])),
                      (round(rectangle[2]), round(rectangle[3])),
                      color=(0, 0, 255), thickness=3)


def draw_plate_polygon(frame, plate_polygon):
    if plate_polygon is not None:
        frame = cv2.polylines(frame, [plate_polygon], 1, (255, 0, 255), 2)
