
def is_checkpoint_passed(car_bbox, checkpoint_line)-> bool:
    return False

def find_nearest_car_to_checkpoint(car_bboxes, checkpoint_line):
    return car_bboxes[0]

def is_plate_number_allowed(detected_plate_number:str):
    return True