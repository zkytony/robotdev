from PIL import Image
import cv2
import torchvision
import torch

def plot_one_box(img, xyxy, label, color, line_thickness=3, show_label=True):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) # line/font thickness
    x1, y1, x2, y2 = map(int, map(round, xyxy))
    cv2.rectangle(img, (x1, y1), (x2, y2), color,
                  thickness=tl, lineType=cv2.LINE_AA)
    if show_label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0,
                                 fontScale=tl/3,
                                 thickness=tf)[0]
        # the background color of the class label
        cv2.rectangle(img,
                      (x1, y1),
                      (x1 + t_size[0], y1 - t_size[1] - 3),
                      color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label,
                    (x1, y1 - 2), 0, tl / 3, [255, 255, 255],
                    thickness=tf, lineType=cv2.LINE_AA)
    return img


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# sourec: https://github.com/matterport/Mask_RCNN/blob/master/samples/demo.ipynb
COCO_CLASS_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def maskrcnn_draw_result(prediction, img,
                         mask_threshold=0.5,
                         class_names=COCO_CLASS_NAMES):
    """
    Let's say you do:

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        img = torchvision.io.read_image(f"./images/{filename}")
        img_input = torch.div(img, 255)
        pred = model([img_input])

    Then you can visualize the result by:
        img = maskrcnn_visualize_result(pred[0], img, ...)

    The output is an ndarray of shape (H, W, 3) with masks, boxes and labels

    Note that 'img' is the original image of shape (3, H, W)
    """
    masks = prediction['masks'].squeeze()
    # need to threshold the mask otherwise will get a box.
    masks = torch.greater(masks, mask_threshold)
    result_img = torchvision.utils.draw_segmentation_masks(img, masks, alpha=0.6)

    boxes = prediction['boxes']
    scores = prediction['scores']
    labels = ["{}({:.2f})".format(class_names[prediction['labels'][i]], scores[i])
              for i in range(len(scores))]
    result_img = torchvision.utils.draw_bounding_boxes(
        result_img, boxes, labels, font_size=15)
    return result_img

def maskrcnn_filter_by_score(prediction, score_threshold=0.7):
    """
    returns segmentation detections with score greather than threhold
    """
    accepted = torch.greater(prediction['scores'], score_threshold)
    result = {}
    result['scores'] = prediction['scores'][accepted]
    result['boxes'] = prediction['boxes'][accepted]
    result['masks'] = prediction['masks'][accepted]
    result['labels'] = prediction['labels'][accepted]
    return result
