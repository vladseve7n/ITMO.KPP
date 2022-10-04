# ITMO.KPP
![Example](misc/test_cam.gif)
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
1. Download train data from https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2021-09-01.zip and unzip it
2. Run train script:
```
cd ocr_model
python train.py
```

### Tensorboard logs
To view tensorboard logs:
1. Download archive https://disk.yandex.ru/d/f3rWwTvhddh0ow
2. Unzip it
3. Run tensorboard