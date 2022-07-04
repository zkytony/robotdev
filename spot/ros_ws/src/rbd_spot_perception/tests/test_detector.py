# Test out detectors
import os
import cv2
import numpy as np
import torch
import torchvision

def test_mask_rcnn():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)

    images = []
    for filename in os.listdir("./images"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = torchvision.io.read_image(f"./images/{filename}")
            img = torch.div(img, 255)
            img = img.cuda(device)
            images.append(img)
    print(f"predicting {len(images)} images...")
    pred = model([images[0]])
    print(pred)


if __name__ == "__main__":
    test_mask_rcnn()
