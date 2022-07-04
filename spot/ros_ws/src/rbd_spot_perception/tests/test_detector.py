# Test out detectors
import os
import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# sourec: https://github.com/matterport/Mask_RCNN/blob/master/samples/demo.ipynb
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def test_mask_rcnn():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    print("Loading model")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)

    # Note that feeding all images to the model at once causes GPU out of memory error.
    # So we have to feed one at a time.
    for filename in os.listdir("./images"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = torchvision.io.read_image(f"./images/{filename}")
            img = torch.div(img, 255)
            img = img.cuda(device)

            print(f"predicting for {filename}...")
            pred = model([img])
            for i in range(len(pred[0]['masks'])):
                mask = pred[0]['masks'][i].permute(1, 2, 0).detach().cpu().numpy()
                label = pred[0]['labels'][i]
                mask = mask.reshape(mask.shape[0], -1)
                plt.imshow(mask, interpolation='none', cmap='gray', vmin=0, vmax=1)
                plt.title(class_names[label.item()])
                plt.show()


if __name__ == "__main__":
    test_mask_rcnn()
