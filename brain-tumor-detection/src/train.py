"""
train.py
---------
This is the MAIN training script. Run this to actually train the model.

What it does, step by step:
  1. Load the full dataset
  2. Split it into training (80%) and validation (20%) sets
  3. Load the model (from model.py)
  4. Loop over the training data for several epochs:
       - show images to the model
       - calculate how wrong it was (loss)
       - adjust the model to be less wrong
  5. After each epoch, save:
       - a checkpoint (the model's current learned state)
       - the loss values (for plotting a graph later)

Run with:
    python src/train.py
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import BrainTumorDataset
from model import get_model


# ------------------------------------------------------------------
# SETTINGS - change these if needed
# ------------------------------------------------------------------
DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"
CHECKPOINT_PATH = os.path.join(OUTPUT_FOLDER, "model_checkpoint.pth")
LOSS_LOG_PATH = os.path.join(OUTPUT_FOLDER, "loss_log.csv")

NUM_EPOCHS = 5          # start small to prove it works, increase later
BATCH_SIZE = 1          # 4GB VRAM limit - keep this at 1
LEARNING_RATE = 0.0001
TRAIN_SPLIT = 0.8        # 80% train, 20% validation


# ------------------------------------------------------------------
# collate_fn
# ------------------------------------------------------------------
def collate_fn(batch):
    """
    Normally PyTorch stacks a batch into one big tensor automatically.
    But our targets (boxes/labels) have different sizes per image,
    so we can't stack them that way.

    This function just keeps each item as-is in a list:
        batch = [(image1, target1), (image2, target2), ...]
    becomes:
        images = [image1, image2, ...]
        targets = [target1, target2, ...]

    Faster R-CNN expects EXACTLY this format: a list of images and
    a list of target dicts.
    """
    return tuple(zip(*batch))


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load dataset ---
    full_dataset = BrainTumorDataset(DATA_FOLDER)
    print(f"Total images: {len(full_dataset)}")

    # --- Train/Validation split ---
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # random_split with a fixed seed = reproducible split
    # (same split every time you run this script)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    # --- DataLoaders ---
    # A DataLoader feeds batches of data to the model during training.
    # shuffle=True for training: mix up the order each epoch so the
    # model doesn't memorize the order of images.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # 0 = simplest, no parallel loading (avoids Windows issues)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # --- Model setup ---
    model = get_model()
    model.to(device)

    # --- Optimizer ---
    # Adam: an optimizer that adapts the learning rate per-parameter.
    # Standard choice for fine-tuning pretrained models.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    # --- Loss log file ---
    # We'll write epoch,train_loss,val_loss to this CSV so we can
    # plot a graph later (Step 6).
    with open(LOSS_LOG_PATH, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")

        # --- TRAINING PHASE ---
        model.train()  # tells the model "we're learning now" (affects some layers)
        total_train_loss = 0.0

        # tqdm wraps the loader to show a progress bar
        for images, targets in tqdm(train_loader, desc="Training"):
            # Move data to GPU
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass: model computes its own loss when given targets
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            # Backward pass: compute gradients and update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        # NOTE: Faster R-CNN's loss calculation requires train() mode.
        # For validation, we still call model() with targets to get a
        # comparable loss number, but we DON'T update the model
        # (no optimizer.step()). We wrap in torch.no_grad() to save memory.
        model.train()
        total_val_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Log losses to CSV ---
        with open(LOSS_LOG_PATH, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

        # --- Save checkpoint after every epoch ---
        # This lets us resume or use the model later without retraining.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
