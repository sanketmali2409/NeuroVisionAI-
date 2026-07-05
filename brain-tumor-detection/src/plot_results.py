"""
plot_results.py
----------------
Reads the loss_log.csv saved during training and plots:
  1. Training loss vs Validation loss across epochs
  2. Saves the plot to outputs/training_curves.png

This is a key visual for your README and portfolio - it proves
your model actually learned something (loss going down) and that
you monitored for overfitting (train vs val comparison).

CONCEPT: OVERFITTING
---------------------
If training loss keeps dropping but validation loss starts going UP,
the model is memorizing the training data instead of learning general
patterns. This is called overfitting. Monitoring both curves is how
you catch this early.

Run with:
    python src/plot_results.py
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


LOSS_LOG_PATH = "outputs/loss_log.csv"
OUTPUT_PATH = "outputs/training_curves.png"


def main():
    if not os.path.exists(LOSS_LOG_PATH):
        print(f"ERROR: {LOSS_LOG_PATH} not found.")
        print("Make sure training has completed and loss_log.csv was saved.")
        return

    # --- Read the CSV ---
    epochs = []
    train_losses = []
    val_losses = []

    with open(LOSS_LOG_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    print(f"Loaded {len(epochs)} epochs of loss data.")
    for e, tl, vl in zip(epochs, train_losses, val_losses):
        print(f"  Epoch {e}: Train Loss = {tl:.4f} | Val Loss = {vl:.4f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, train_losses, marker="o", linewidth=2,
            color="#2196F3", label="Training Loss")
    ax.plot(epochs, val_losses, marker="s", linewidth=2,
            color="#F44336", label="Validation Loss")

    # Labels and formatting
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Brain Tumor Detection — Training vs Validation Loss", fontsize=13)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)

    print(f"\nPlot saved to: {OUTPUT_PATH}")
    print("Add this image to your README under 'Training Results'.")

    # --- Quick analysis ---
    print("\n--- Quick Analysis ---")
    if train_losses[-1] < train_losses[0]:
        print("✅ Training loss decreased — model is learning.")
    else:
        print("⚠️  Training loss did not decrease — check learning rate or data.")

    if val_losses[-1] < val_losses[0]:
        print("✅ Validation loss decreased — model generalizes well.")
    else:
        print("⚠️  Validation loss increased — possible overfitting.")
        print("   Consider: more epochs, data augmentation, or lower learning rate.")


if __name__ == "__main__":
    main()
