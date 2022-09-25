"""
python3 -m nomeroff_net.text_detectors.ru -f nomeroff_net/text_detectors/ru.py
"""
import torch
from .base.ocr import OCR
from nomeroff_net.tools.mcm import get_device_torch


class Ru(OCR):
    def __init__(self) -> None:
        OCR.__init__(self)
        self.letters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "E", "H", "K", "M", "O",
                        "P", "T", "X", "Y"]
        self.max_text_len = 9
        self.max_plate_length = 9
        self.letters_max = len(self.letters)+1
        self.init_label_converter()


ru = Ru

if __name__ == "__main__":
    ocr = Ru()
    ocr.load()
    device = get_device_torch()
    xs = torch.rand((1, 3, 50, 200)).to(device)
    y = ocr.predict(xs)
    print(y)
