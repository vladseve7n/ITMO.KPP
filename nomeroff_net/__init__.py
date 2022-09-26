from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points import NpPointsCraft
from nomeroff_net.pipes.number_plate_text_readers.text_detector import TextDetector
from nomeroff_net.pipes.number_plate_text_readers.text_postprocessing import (text_postprocessing,
                                                                              text_postprocessing_async)
from nomeroff_net.pipes.number_plate_classificators.options_detector import OptionsDetector
from nomeroff_net.pipes.number_plate_classificators.inverse_detector import InverseDetector
from nomeroff_net.pipes.number_plate_localizators.yolo_v5_detector import Detector

from nomeroff_net.pipelines import pipeline


__version__ = "3.2.2"
