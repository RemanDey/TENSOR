"""
    python trainer.py --csv_path /path/to/train_labels.csv --epochs 10 --batch_size 16
"""

import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ===============================
#  CONFIGURATION
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ===============================
#  DATASET CLASS
# ===============================
class ForgeryDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        # Normalize label values to 0 (fake) / 1 (real)
        self.data["label"] = self.data["label"].str.lower().map({"fake": 0, "real": 1})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["image_path"]
        label = int(self.data.iloc[idx]["label"])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

# ===============================
#  MODEL DEFINITION
# ===============================
class ForgeryDetector(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(pretrained=True)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base.classifier[1].in_features, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = torch.sigmoid(self.fc(x))
        return x

# ===============================
#  TRAINING FUNCTION
# ===============================
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, save_path="model.pt"):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses, train_preds, train_labels = [], [], []

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = (outputs > 0.5).int().cpu().numpy().flatten()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy().flatten())

        train_acc = accuracy_score(train_labels, train_preds)
        train_loss = np.mean(train_losses)

        # ------------------------------
        #  Validation
        # ------------------------------
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(imgs)
                preds = (outputs > 0.5).int().cpu().numpy().flatten()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy().flatten())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        print(f" Epoch {epoch}: TrainLoss={train_loss:.4f} | TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f} | ValF1={val_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model → {save_path}")

    print("Training complete. Best Val Accuracy:", best_val_acc)

# ===============================
# MAIN FUNCTION
# ===============================
def main(args):
    # Data augmentations
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load full dataset
    full_dataset = ForgeryDataset(args.csv_path, transform=train_tfms)

    # Split into train/val (90/10)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Use simple ToTensor on validation
    val_ds.dataset.transform = val_tfms

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize model
    model = ForgeryDetector().to(device)

    # Train
    train_model(model, train_loader, val_loader,
                epochs=args.epochs, lr=args.lr, save_path=args.save_path)

# ===============================

# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to train_labels.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="model.pt")
    args = parser.parse_args()

    main(args)
