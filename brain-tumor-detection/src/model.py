"""
model.py
---------
This file has ONE function: get_model().

It loads a pretrained Faster R-CNN object detection model and adapts
it to our problem: detecting 3 types of brain tumors.

CONCEPT: TRANSFER LEARNING
---------------------------
Instead of training a neural network from zero (which needs millions
of images and huge compute), we start with a model ALREADY trained
on a massive dataset (COCO - 330,000 everyday photos with 91 object
categories like "dog", "car", "person").

That pretrained model already learned how to:
  - detect edges, shapes, textures (early layers)
  - find "objects" in general (middle layers)

We KEEP all of that. We only replace the very LAST layer - the part
that decides "WHICH category is this object?" - because COCO's 91
categories (dog, car...) are useless for us. We swap it for OUR
4 categories instead.

WHY RESNET-50 (not MobileNet)?
--------------------------------
You have an NVIDIA RTX 3050 (4GB VRAM) - enough to train ResNet-50
based Faster R-CNN with a small batch size. ResNet-50+FPN is the
more accurate, more "standard" architecture and is a stronger line
on your resume: "Faster R-CNN with ResNet-50 + FPN backbone".
"""

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Our classes:
# 0 = background (Faster R-CNN ALWAYS needs a "background" class at index 0)
# 1 = Meningioma
# 2 = Glioma
# 3 = Pituitary
NUM_CLASSES = 4


def get_model(num_classes=NUM_CLASSES):
    """
    Returns a Faster R-CNN (ResNet-50 + FPN) model ready for training
    on our dataset.
    """

    # Step 1: Load the pretrained model (trained on COCO dataset)
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Step 2: Find out how many input features the classification head expects.
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Step 3: Replace the head with a new one sized for OUR number of classes.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ------------------------------------------------------------------
# Quick test - run this file directly:
#     python src/model.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = get_model()
    model.to(device)
    model.eval()

    # Create one fake image (3 channels, 512x512) to test the model runs
    dummy_image = torch.rand(3, 512, 512).to(device)

    with torch.no_grad():
        output = model([dummy_image])

    print("Model loaded successfully.")
    print("Output keys:", output[0].keys())
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
