"""
evaluate.py
------------
Computes mAP (mean Average Precision) and IoU-based metrics on the
VALIDATION set, using the trained model checkpoint.

WHY THIS MATTERS:
This was the #1 missing piece in the original project. "Accuracy"
alone doesn't mean much for object detection - mAP is the metric
recruiters and standard benchmarks (COCO, Pascal VOC) actually use.

CONCEPTS:
  - IoU (Intersection over Union): how much a predicted box overlaps
    with the real (ground truth) box. 1.0 = perfect match, 0 = no overlap.
  - mAP (mean Average Precision): averages precision across all
    classes and IoU thresholds. The standard object detection metric.

We use the `torchmetrics` library's MeanAveragePrecision - this is
the same implementation used in official benchmarks, so the numbers
are trustworthy and comparable to published results.

Run with:
    python src/evaluate.py
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset import BrainTumorDataset
from model import get_model
from train import collate_fn, DATA_FOLDER, CHECKPOINT_PATH, BATCH_SIZE, TRAIN_SPLIT


LABEL_NAMES = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load dataset and recreate the SAME validation split as training ---
    # IMPORTANT: same seed (42) as train.py, so we evaluate on images the
    # model did NOT train on.
    full_dataset = BrainTumorDataset(DATA_FOLDER)
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Validation images: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # --- Load model + trained weights ---
    model = get_model()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()  # evaluation mode: model just predicts, no loss/training
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # --- Set up the mAP metric ---
    # class_metrics=True gives us a per-class breakdown too, not just an
    # overall average.
    metric = MeanAveragePrecision(class_metrics=True)

    # --- Run predictions on validation set ---
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]

            # Get model's predictions (boxes, labels, confidence scores)
            predictions = model(images)

            # Move everything to CPU for the metric calculation
            predictions = [{k: v.cpu() for k, v in p.items()} for p in predictions]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

            # Feed this batch's predictions + ground truth to the metric
            metric.update(predictions, targets_cpu)

    # --- Compute final results ---
    results = metric.compute()

    print("\n========== EVALUATION RESULTS ==========")
    print(f"mAP (IoU 0.50:0.95): {results['map'].item():.4f}")
    print(f"mAP @ IoU=0.50:      {results['map_50'].item():.4f}")
    print(f"mAP @ IoU=0.75:      {results['map_75'].item():.4f}")
    print(f"Mean Recall:         {results['mar_100'].item():.4f}")

    # --- Per-class breakdown ---
    if "map_per_class" in results:
        print("\n--- Per-Class mAP (IoU 0.50:0.95) ---")
        per_class = results["map_per_class"]
        classes_present = results["classes"]
        for cls_id, score in zip(classes_present.tolist(), per_class.tolist()):
            name = LABEL_NAMES.get(cls_id, f"Class {cls_id}")
            print(f"  {name}: {score:.4f}")

    print("==========================================")

    # --- Save results to a text file for the README ---
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_results.txt", "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("===================\n")
        f.write(f"mAP (IoU 0.50:0.95): {results['map'].item():.4f}\n")
        f.write(f"mAP @ IoU=0.50:      {results['map_50'].item():.4f}\n")
        f.write(f"mAP @ IoU=0.75:      {results['map_75'].item():.4f}\n")
        f.write(f"Mean Recall:         {results['mar_100'].item():.4f}\n")

    print("\nResults also saved to outputs/evaluation_results.txt")


if __name__ == "__main__":
    main()
