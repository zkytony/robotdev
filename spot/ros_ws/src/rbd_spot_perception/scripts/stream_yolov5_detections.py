import torch
import os

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yoloV5Weights/best.pt')
img = 'yoloV5Weights/test.JPG'

results = model(img)

results.print()
# print("<<<>>>")
yoloPandaDataframe= results.pandas().xyxy[0]

for index, row in yoloPandaDataframe.iterrows():
    print('name= ', row['name'], 'score=',row['confidence'],'x1=',int(row['xmin']), 'y1=',int(row['ymin']),'x2=',int(row['xmax']), 'y2=',int(row['ymax']))
    print(type(row['name']))
    print(type(row['confidence']))
    print(type(row['xmin']))