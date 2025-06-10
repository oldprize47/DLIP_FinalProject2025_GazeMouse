# 파일명: evaluate_eyes.py
import json
import math
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from eye_dataset_file import RTGENDatasetEyes
from fginet_eyes import FGINetEyes


def rad2deg(rad):
    return rad * (180.0 / math.pi)


# 설정
CKPT = "best_fgineteyes.pth"
SPLIT_JS = "split_indices.json"
CSV_PATH = "RT_GENE/rtgene_pairs_face_eye.csv"
BATCH = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 데이터 및 모델 로드
with open(SPLIT_JS, "r") as f:
    splits = json.load(f)
test_idx = splits["test"]

full_ds = RTGENDatasetEyes(CSV_PATH)
test_ds = Subset(full_ds, test_idx)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

model = FGINetEyes().to(device)
try:
    # PyTorch ≥2.4 에서만 weights_only 지원
    sd = torch.load(CKPT, map_location=device, weights_only=True)
except TypeError:
    # 구버전 PyTorch 호환용
    sd = torch.load(CKPT, map_location=device)
model.load_state_dict(sd)
model.eval()

# 2) 평가
errors = []  # list of [h_yaw_err, h_pitch_err, g_yaw_err, g_pitch_err]
with torch.no_grad():
    for face, eyes, labels in test_loader:
        face, eyes = face.to(device), eyes.to(device)
        preds = model(face, eyes).cpu().numpy()
        gts = labels.numpy()
        # 절대오차 (rad)
        abs_err = np.abs(preds - gts)
        errors.append(abs_err)

errors = np.vstack(errors)  # (N,4)
# MAE (rad)
mae_rad = errors.mean(axis=0)
# MAE (deg)
mae_deg = rad2deg(mae_rad)

print("Test set MAE (rad):")
print(f"  Head yaw   : {mae_rad[0]:.4f}")
print(f"  Head pitch : {mae_rad[1]:.4f}")
print(f"  Gaze yaw   : {mae_rad[2]:.4f}")
print(f"  Gaze pitch : {mae_rad[3]:.4f}")
print("Test set MAE (deg):")
print(f"  Head yaw   : {mae_deg[0]:.2f}°")
print(f"  Head pitch : {mae_deg[1]:.2f}°")
print(f"  Gaze yaw   : {mae_deg[2]:.2f}°")
print(f"  Gaze pitch : {mae_deg[3]:.2f}°")
overall_mae_deg = mae_deg.mean()
print(f"Overall MAE: {overall_mae_deg:.2f}°")
