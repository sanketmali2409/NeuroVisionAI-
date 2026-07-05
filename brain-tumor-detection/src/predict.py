"""
predict.py
-----------
Loads the trained model and runs it on ONE image from the dataset.
Saves a picture showing:
  - the MRI scan
  - the GROUND TRUTH box (green) - the real tumor location
  - the PREDICTED box (red) - what the model thinks
  - the predicted class name + confidence score

This is the script that PROVES your model works - great for README,
demos, and interviews ("let me show you it running on a new image").

CONCEPT: CONFIDENCE THRESHOLD
-------------------------------
The model outputs many candidate boxes, each with a "score" (0 to 1)
representing how confident it is. We only keep predictions above
CONFIDENCE_THRESHOLD (e.g. 0.5) - otherwise the image gets cluttered
with low-confidence guesses.

Run with:
    python src/predict.py --index 5

(--index = which image in the dataset to test, default 0)
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset import BrainTumorDataset
from model import get_model
from train import DATA_FOLDER, CHECKPOINT_PATH


LABEL_NAMES = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}
CONFIDENCE_THRESHOLD = 0.5
OUTPUT_FOLDER = "outputs/sample_predictions"


def main(index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load dataset and pick ONE image ---
    dataset = BrainTumorDataset(DATA_FOLDER)
    image_tensor, target = dataset[index]
    print(f"Loaded image #{index} from dataset (total: {len(dataset)})")

    # --- Load model + trained weights ---
    model = get_model()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # --- Run prediction ---
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])[0]

    # prediction has 'boxes', 'labels', 'scores' - all sorted by
    # confidence (highest first) automatically by Faster R-CNN.
    pred_boxes = prediction["boxes"].cpu()
    pred_labels = prediction["labels"].cpu()
    pred_scores = prediction["scores"].cpu()

    # --- Filter by confidence threshold ---
    keep = pred_scores >= CONFIDENCE_THRESHOLD
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    print(f"Predictions above {CONFIDENCE_THRESHOLD} confidence: {len(pred_boxes)}")

    # --- Plot ---
    image_np = image_tensor.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image_np)

    # Ground truth box in GREEN
    gt_box = target["boxes"][0].numpy()
    gt_label = int(target["labels"][0])
    gx, gy, gx2, gy2 = gt_box
    rect_gt = patches.Rectangle(
        (gx, gy), gx2 - gx, gy2 - gy,
        linewidth=2, edgecolor="lime", facecolor="none", label="Ground Truth"
    )
    ax.add_patch(rect_gt)

    # Predicted box(es) in RED
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = box.numpy()
        rect_pred = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none", label="Prediction"
        )
        ax.add_patch(rect_pred)

        label_name = LABEL_NAMES.get(int(label), f"Class {int(label)}")
        ax.text(
            x1, max(y1 - 5, 0),
            f"{label_name} ({score:.2f})",
            color="red", fontsize=10, weight="bold",
            bbox=dict(facecolor="white", alpha=0.7, pad=1)
        )

    gt_name = LABEL_NAMES.get(gt_label, f"Class {gt_label}")
    ax.set_title(f"Image #{index} | Ground Truth: {gt_name} (green)")
    ax.axis("off")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    save_path = os.path.join(OUTPUT_FOLDER, f"prediction_{index}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved result to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Index of image in dataset to test")
    args = parser.parse_args()
    main(args.index)
