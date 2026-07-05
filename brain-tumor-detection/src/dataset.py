"""
dataset.py
-----------
This file defines ONE class: BrainTumorDataset.

Its job: given a folder full of .mat files, let PyTorch read each one
as (image, target) where:
    image  = the MRI scan as a tensor
    target = a dictionary with the tumor's bounding box and class label

This is the ONLY dataset class in the project. (The old notebooks had
3 different versions - we don't need that anymore.)
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class BrainTumorDataset(Dataset):
    def __init__(self, folder):
        """
        This runs ONCE when you create the dataset.
        It just looks at the folder and makes a list of all .mat files.

        folder: path to the folder containing your ~3000 .mat files
        """
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".mat")])

    def __len__(self):
        """
        PyTorch calls this to know how many images exist in total.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        PyTorch calls this with an index (idx) like 0, 1, 2, ... up to len()-1.
        We must return ONE (image, target) pair.

        Steps:
        1. Open the .mat file at position `idx`
        2. Read the MRI image, the tumor mask, and the label
        3. Convert the mask into a bounding box (a rectangle)
        4. Convert everything into PyTorch tensors
        5. Package it all into the format Faster R-CNN expects
        """
        file_path = os.path.join(self.folder, self.files[idx])

        # --- Step 1 & 2: open the .mat file (HDF5 / MATLAB v7.3 format) ---
        with h5py.File(file_path, "r") as f:
            cjdata = f["cjdata"]
            image = np.array(cjdata["image"]).T          # MRI scan, 2D grid of numbers
            mask = np.array(cjdata["tumorMask"]).T        # 0/1 grid showing tumor location
            label = int(np.array(cjdata["label"])[0][0])  # 1, 2, or 3 (tumor type)

        # --- Step 3: normalize image pixel values to range [0, 1] ---
        # Why? Neural networks train better when input numbers are small
        # and consistent. Raw MRI pixel values can be in weird ranges
        # (e.g. 0-4000), so we squash them to 0-1.
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        # the "+ 1e-8" prevents a divide-by-zero error on blank images

        # --- Step 4: convert grayscale (1 channel) to RGB-like (3 channels) ---
        # Why? Faster R-CNN with a ResNet-50 backbone expects 3-channel
        # input (like normal color photos). Our MRI is grayscale (1 channel),
        # so we just duplicate it 3 times. This is the standard trick.
        image_3channel = np.stack([image, image, image], axis=-1)  # shape: (H, W, 3)

        # Convert to a PyTorch tensor with shape (3, H, W) - PyTorch wants
        # channels FIRST, not last.
        image_tensor = F.to_tensor(image_3channel)

        # --- Step 5: convert the tumor MASK (blob) into a BOUNDING BOX (rectangle) ---
        # np.where finds all the pixel coordinates where mask == 1 (i.e. tumor pixels)
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            # Safety fallback: if a mask is somehow empty, use the whole image
            # as the box. This avoids crashes on bad data.
            xmin, ymin, xmax, ymax = 0, 0, mask.shape[1], mask.shape[0]
        else:
            xmin, xmax = float(xs.min()), float(xs.max())
            ymin, ymax = float(ys.min()), float(ys.max())

        # Faster R-CNN expects boxes as a tensor of shape (num_boxes, 4)
        # We only have 1 tumor per image, so num_boxes = 1
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

        # Labels must also be a tensor, shape (num_boxes,)
        labels = torch.tensor([label], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image_tensor, target


# ------------------------------------------------------------------
# Quick test - run this file directly to check everything works:
#     python src/dataset.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    DATA_FOLDER = "data"  # adjust if needed

    dataset = BrainTumorDataset(DATA_FOLDER)
    print(f"Total images found: {len(dataset)}")

    image, target = dataset[0]
    print("Image shape:", image.shape)      # should be (3, H, W)
    print("Image dtype:", image.dtype)
    print("Image min/max:", image.min().item(), image.max().item())
    print("Bounding box:", target["boxes"])
    print("Label:", target["labels"])
