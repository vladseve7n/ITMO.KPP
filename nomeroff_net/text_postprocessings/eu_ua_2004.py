"""
python3 -m nomeroff_net.text_postprocessings.eu_ua_2004 -f nomeroff_net/text_postprocessings/eu_ua_2004.py
"""
import operator
from collections import Counter
from .xx_xx import XxXx


class EuUa2004(XxXx):
    def __init__(self) -> None:
        super().__init__()
        self.STANDART = "@@####@@"
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
        self.STAT = {
             'II': 29,
             'AI': 314728,
             'AA': 596728,
             'AB': 204175,
             'AE': 382942,
             'AK': 58144,
             'AM': 154709,
             'AH': 282251,
             'AO': 110668,
             'AP': 208508,
             'AC': 136755,
             'AT': 159856,
             'AX': 294454,
             'BI': 189327,
             'BA': 129620,
             'BB': 125643,
             'BE': 144443,
             'BK': 145141,
             'BM': 121848,
             'BH': 342605,
             'BO': 121445,
             'BC': 282600,
             'BT': 120082,
             'BX': 157944,
             'CA': 164098,
             'CB': 129214,
             'CE': 104713,
             'CH': 16935,
             'EE': 1,
             'PB': 0,
             'PK': 0,
             'EO': 0,
             'TB': 0,
             'KA': 410,
             'KE': 154,
             'KX': 198,
             'HE': 0,
             'HH': 71,
             'TX': 0,
             'KI': 4,
             'KH': 0
        }

    def do_stat_anal(self, text: str) -> str:
        if len(text) < 2 or text[:2] in self.STAT:
            return text
        v = Counter()
        for item in self.STAT.keys():
            if item[0] == text[0] or item[1] == text[1]:
                v[item] += self.STAT[item]

        if bool(v):
            region_key = max(v.items(), key=operator.itemgetter(1))[0]
            return "{}{}".format(region_key, text[2:])
        return text

    def find(self, text: str, strong: bool = False) -> str:
        text = super().find(text, strong)
        return self.do_stat_anal(text)


eu_ua_2004 = EuUa2004()

if __name__ == "__main__":
    postprocessor = EuUa2004()
    assert postprocessor.find("AB1234CH1") == "AB1234CH"
