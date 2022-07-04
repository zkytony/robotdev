# Test out detectors
import os
import cv2
import numpy as np
import torch
import torchvision
from rbd_spot_perception.utils.vision.detector import (maskrcnn_draw_result,
                                                       maskrcnn_filter_by_score)
import matplotlib.pyplot as plt

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
            img_input = torch.div(img, 255)
            img_input = img_input.cuda(device)

            print(f"predicting for {filename}...")
            pred = model([img_input])[0]
            pred = maskrcnn_filter_by_score(pred)
            result_img = maskrcnn_draw_result(pred, img)
            plt.imshow(result_img.permute(1, 2, 0), interpolation='none')
            plt.show()



if __name__ == "__main__":
    test_mask_rcnn()
