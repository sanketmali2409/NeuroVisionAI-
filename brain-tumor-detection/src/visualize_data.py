"""
visualize_data.py
-------------------
Purpose: SANITY CHECK ONLY. This script does NOT train anything.

It loads a few images from BrainTumorDataset, draws the bounding box
on each one, and saves the result as a .png file so you can LOOK at it
and confirm:
    1. The image looks like a real brain MRI
    2. The bounding box actually sits on top of the tumor

If the boxes look wrong here, DO NOT proceed to training - tell me
and we'll debug the dataset.py file first.

Run with:
    python visualize_data.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import BrainTumorDataset

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs/sample_predictions"
NUM_SAMPLES = 5  # how many random images to check

# A human-readable name for each label number, based on your dataset description
LABEL_NAMES = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    dataset = BrainTumorDataset(DATA_FOLDER)
    print(f"Dataset loaded. Total images: {len(dataset)}")

    for i in range(NUM_SAMPLES):
        image_tensor, target = dataset[i]

        # image_tensor has shape (3, H, W). For displaying, matplotlib
        # wants (H, W, 3) - so we rearrange the dimensions.
        # .permute(1, 2, 0) means: "move dimension 0 to the end"
        image_np = image_tensor.permute(1, 2, 0).numpy()

        box = target["boxes"][0].numpy()  # [xmin, ymin, xmax, ymax]
        label_num = int(target["labels"][0])
        label_name = LABEL_NAMES.get(label_num, f"Unknown({label_num})")

        # --- Plotting ---
        fig, ax = plt.subplots(1, figsize=(5, 5))
        ax.imshow(image_np)

        # Draw the bounding box as a red rectangle
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        ax.set_title(f"Sample {i} | Label: {label_name}")
        ax.axis("off")

        save_path = os.path.join(OUTPUT_FOLDER, f"sample_{i}_check.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {save_path}  |  Label: {label_name}  |  Box: {box}")

    print("\nDone. Open the images in outputs/sample_predictions/ and check:")
    print("  1. Does it look like a brain MRI?")
    print("  2. Is the red box actually around the bright tumor blob?")


if __name__ == "__main__":
    main()
