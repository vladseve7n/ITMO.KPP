import cv2


def draw_textlines(frame, textlines:list, x_start=10, y_start=70):
    i_increment = 35
    for i, line in enumerate(textlines):
        y = i*i_increment + y_start
        cv2.putText(img=frame, text=line, org=(x_start, y), 
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)


def draw_checkpoint_line(frame, line_coords):
    cv2.line(frame,line_coords[0],line_coords[1],(255,0,0),5)


def draw_rectangles(frame, rectangles):
    for rectangle in rectangles:
        #print((round(rectangle[0]), round(rectangle[1])),
        #    (round(rectangle[2]), round(rectangle[3])))
        cv2.rectangle(frame,(round(rectangle[0]), round(rectangle[1])),
            (round(rectangle[2]), round(rectangle[3])),
            color=(0,0,255),thickness=3)