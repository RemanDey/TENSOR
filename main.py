"""
ZenTej Season 3 Hackathon - Deepfake-Proof eKYC Challenge
Final Inference Script (Compatible with Evaluation Format)

Usage:
    python main.py --test_path /content/test --labels_csv /content/labels_test.csv
"""

import os
import csv
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report

# ================================
#  CONFIGURATION (Leave Blank)
# ================================
TEST_DATASET_PATH = ""
LABELS_CSV_PATH = "train_labels.csv"

# ================================
#  MODEL DEFINITIONS
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Face preprocessing and embedding model (for identity verification)
mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
face_encoder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Forgery detector architecture (must match training)
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
#  LOAD TRAINED MODEL
# ================================
def load_model():
    forgery_model = ForgeryDetector().to(device)
    forgery_model.load_state_dict(torch.load("model.pt", map_location=device))
    forgery_model.eval()
    print(" Loaded trained forgery model successfully.")
    return forgery_model

# ================================
#  HELPER FUNCTIONS
# ================================
transform_forgery = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess_face(pil_img):
    face = mtcnn(pil_img)
    if face is None:
        return None
    return face.unsqueeze(0).to(device)

def get_embedding(pil_img):
    face = preprocess_face(pil_img)
    if face is None:
        return None
    with torch.no_grad():
        emb = face_encoder(face)
    return emb.cpu().numpy()

def compute_match_score(img1, img2):
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)
    if emb1 is None or emb2 is None:
        return 0.0
    score = cosine_similarity(emb1, emb2)[0][0]
    return float(score)

def compute_liveness_score(img, model):
    x = transform_forgery(img).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(x).item()
    return float(score)

# ================================
#  INFERENCE FOR ONE PAIR
# ================================
def predict_match_and_fake(id_image, selfie_image, model):
    """
    Returns:
        (is_match: 0/1, is_fake: 0/1)
    """
    match_score = compute_match_score(id_image, selfie_image)
    liveness_id = compute_liveness_score(id_image, model)
    liveness_selfie = compute_liveness_score(selfie_image, model)
    avg_live = (liveness_id + liveness_selfie) / 2

    # thresholds (tuned experimentally)
    is_match = 1 if match_score >= 0.6 else 0
    is_fake = 0 if avg_live >= 0.5 else 1

    return is_match, is_fake

# ================================
#  FULL INFERENCE LOOP
# ================================
def run_inference():
    model = load_model()
    predictions = []

    for kyc_folder in sorted(os.listdir(TEST_DATASET_PATH)):
        folder_path = os.path.join(TEST_DATASET_PATH, kyc_folder)
        if not os.path.isdir(folder_path):
            continue

        id_path = os.path.join(folder_path, "id.jpg")
        if not os.path.exists(id_path):
            continue

        id_image = Image.open(id_path).convert("RGB")

        for file in os.listdir(folder_path):
            if file.startswith("selfie_"):
                selfie_image = Image.open(os.path.join(folder_path, file)).convert("RGB")
                is_match, is_fake = predict_match_and_fake(id_image, selfie_image, model)
                predictions.append([kyc_folder, file, is_match, is_fake])

    return predictions

# ================================
#  EVALUATION
# ================================
def evaluate(predictions):
    gt = {}
    with open(LABELS_CSV_PATH, "r") as f:
        reader = csv.reader(f)
        header = next(reader) if "ID" in next(csv.reader(open(LABELS_CSV_PATH))).__str__() else None
        f.seek(0)
        for row in reader:
            # Expected format: ID, selfie_name, match_label, fake_label
            gt[(row[0], row[1])] = (int(row[2]), int(row[3]))

    y_true_match, y_pred_match = [], []
    y_true_fake, y_pred_fake = [], []

    for (kyc, selfie, pm, pf) in predictions:
        if (kyc, selfie) in gt:
            tm, tf = gt[(kyc, selfie)]
            y_true_match.append(tm)
            y_pred_match.append(pm)
            y_true_fake.append(tf)
            y_pred_fake.append(pf)

    print("===== MATCHING METRICS =====")
    print(classification_report(y_true_match, y_pred_match, digits=4))

    print("===== FAKE DETECTION METRICS =====")
    print(classification_report(y_true_fake, y_pred_fake, digits=4))

# ================================
#  MAIN EXECUTION
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default="", help="Path to test dataset")
    parser.add_argument("--labels_csv", type=str, default="", help="Path to labels CSV")
    args = parser.parse_args()

    TEST_DATASET_PATH = args.test_path
    LABELS_CSV_PATH = args.labels_csv

    if not TEST_DATASET_PATH or not LABELS_CSV_PATH:
        raise ValueError("Please specify --test_path and --labels_csv")

    preds = run_inference()
    evaluate(preds)
