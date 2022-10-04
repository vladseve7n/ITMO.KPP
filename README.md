# ITMO.KPP
This project simulates automated road checkpoint. 
To run project on a test video:
1. Download checkpoint from https://disk.yandex.ru/d/AhYyZ98ahahcVA and place it in 'misc' folder
2. Install requirements:
```
pip install -r requirements.txt
```
3. Run main script:
```
python main.py
```

### Training
To train OCR model:
1. Download train data and unzip it
2. Run train script:
```
cd ocr_model
python train.py
```