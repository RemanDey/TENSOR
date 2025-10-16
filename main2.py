
import os
import csv
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from sklearn.metrics import classification_report

# ================================
# CONFIGURATION
# ================================
TEST_DATASET_PATH = ""
LABELS_CSV_PATH = ""
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================================
# MODEL DEFINITION (same as training)
# ================================
class ForgeryDetector(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(pretrained=False)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base.classifier[1].in_features, 1)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = torch.sigmoid(self.fc(x))
        return x

# ================================
# LOAD MODEL
# ================================
def load_model():
    model = ForgeryDetector().to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()
    print("✅ Loaded trained forgery model successfully.")
    return model

# ================================
# PREPROCESSING
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ================================
# PREDICTION
# ================================
def predict_fake_or_real(image_path, model):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(x).item()
    label_pred = "real" if score >= 0.5 else "fake"
    return label_pred, score

# ================================
# FULL INFERENCE LOOP
# ================================
def run_inference():
    model = load_model()
    predictions = []

    for folder in sorted(os.listdir(TEST_DATASET_PATH)):
        folder_path = os.path.join(TEST_DATASET_PATH, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if not (file.lower().endswith(".jpg") or file.lower().endswith(".png")):
                continue

            img_path = os.path.join(folder_path, file)
            label_pred, score = predict_fake_or_real(img_path, model)
            predictions.append([os.path.join(folder, file), label_pred, score])

    return predictions

# ================================
# EVALUATION
# ================================
def evaluate(predictions):
    gt = {}
    with open(LABELS_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row["image_path"]] = row["label"].lower()

    y_true, y_pred = [], []
    for (img_relpath, pred_label, _) in predictions:
        if img_relpath in gt:
            y_true.append(gt[img_relpath])
            y_pred.append(pred_label)

    print("===== FAKE DETECTION METRICS =====")
    print(classification_report(y_true, y_pred, digits=4))

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True, help="Path to test dataset root")
    parser.add_argument("--labels_csv", type=str, required=True, help="Path to labels CSV (image_path,label)")
    args = parser.parse_args()

    TEST_DATASET_PATH = args.test_path
    LABELS_CSV_PATH = args.labels_csv

    preds = run_inference()
    evaluate(preds)
