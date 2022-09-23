import cv2


def place_info_on_frame(frame, textlines:list, x_start=10, y_start=70):
    i_increment = 30
    for i, line in enumerate(textlines):
        y = i*i_increment + y_start
        cv2.putText(img=frame, text=line, org=(x_start, y), 
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)