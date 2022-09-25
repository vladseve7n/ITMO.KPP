"""
python3 -m nomeroff_net.text_postprocessings.eu_ua_2004_squire -f nomeroff_net/text_postprocessings/eu_ua_2004_squire.py
"""
from .xx_xx import XxXx


class EuUa2004Squire(XxXx):
    def __init__(self) -> None:
        super().__init__()
        self.STANDART = "@@@@####"
        self.ALLOWED_LITERS = ["A", "B", "E", "I", "K", "M", "H", "O", "P", "C", "T", "X"]
        self.REPLACEMENT = {
            "#": {
                "O": "0",
                "Q": "0",
                "D": "0",
                "I": "1",
                "Z": "2",  # 7
                "S": "5",  # 8
                "T": "7",
                "B": "8"
            },
            "@": {
                "/": "I",
                "|": "I",
                "L": "I",
                "1": "I",
                "5": "B",
                "8": "B",
                "R": "B",
                "0": "O",
                "Q": "O",
                "¥": "X",
                "Y": "X",
                "€": "C",
                "F": "E"
            }
        }

    def find(self, text: str, strong: bool = False) -> str:
        text = super().find(text, strong)
        if len(text) == 8:
            text = text[:2] + text[4:8] + text[2:4]
        return text


eu_ua_2004_squire = EuUa2004Squire()

if __name__ == "__main__":
    postprocessor = EuUa2004Squire()
    print(postprocessor.find("ABHH1234"))
