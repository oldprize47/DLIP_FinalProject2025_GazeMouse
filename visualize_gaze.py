# 파일명: visualize_gaze.py
import json, random, math, cv2, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from torchvision import transforms
from eye_dataset_file import RTGENDatasetEyes
from fginet_eyes import FGINetEyes

# 설정
CSV_PATH = "RT_GENE/rtgene_pairs_face_eye.csv"
SPLIT_JS = "split_indices.json"
CKPT = "best_fgineteyes.pth"
NUM_SHOW = 4  # 표시할 이미지 수

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model = FGINetEyes().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True if hasattr(torch.load, "weights_only") else {}))
model.eval()

# CSV와 split 정보 로드
df = pd.read_csv(CSV_PATH)
splits = json.load(open(SPLIT_JS, "r"))
test_idx = splits["test"]

# yaw/pitch 최대값으로 정규화용
max_yaw = df["gaze_yaw"].abs().max()
max_pitch = df["gaze_pitch"].abs().max()

# 전처리 정의
face_tf = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

eye_tf = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 무작위 샘플 선택
samples = random.sample(test_idx, NUM_SHOW)

# layout
cols = 2
rows = math.ceil(NUM_SHOW / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
axes = axes.flatten()


# 화살표 그리기 헬퍼
def draw_direction(img, yaw, pitch, color, max_yaw, max_pitch, length=5000, thickness=2):
    """
    • yaw  (+) = 얼굴이 왼쪽 → 화살표 왼쪽(-x)
    • pitch(+) = 얼굴이 아래   → 화살표 아래(+y)
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # 화면 픽셀 단위 오프셋
    dx = -length * math.sin(math.radians(yaw))  # x (좌/우)
    dy = -length * math.sin(math.radians(pitch))  # y (위/아래)

    cv2.arrowedLine(img, (cx, cy), (int(cx + dx), int(cy + dy)), color=color, thickness=thickness, tipLength=0.25)


# 시각화
for i, idx in enumerate(samples):
    row = df.iloc[idx]
    img = cv2.imread(row["face_path"])
    disp = img.copy()

    # 모델 입력 구성
    face = face_tf(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    le = eye_tf(cv2.imread(row["left_eye_path"])[:, :, ::-1]).unsqueeze(0).to(device)
    re = eye_tf(cv2.imread(row["right_eye_path"])[:, :, ::-1]).unsqueeze(0).to(device)
    eyes = torch.cat([le, re], dim=1)

    # 예측값
    with torch.no_grad():
        pred = model(face, eyes).cpu().numpy().flatten()
    hy_pred, hp_pred, gy_pred, gp_pred = pred[0], pred[1], pred[2], pred[3]
    hy_gt, hp_gt, gy_gt, gp_gt = row["head_yaw"], row["head_pitch"], row["gaze_yaw"], row["gaze_pitch"]

    # 머리 방향 (초록 = 예측, 진초록 = GT)
    draw_direction(disp, hy_pred, hp_pred, (0, 255, 0), max_yaw, max_pitch)
    draw_direction(disp, hy_gt, hp_gt, (0, 200, 0), max_yaw, max_pitch)
    # 시선 방향 (빨강 = 예측, 진빨강 = GT)
    draw_direction(disp, gy_pred, gp_pred, (0, 0, 255), max_yaw, max_pitch)
    draw_direction(disp, gy_gt, gp_gt, (0, 0, 200), max_yaw, max_pitch)

    # 결과 표시
    axes[i].imshow(disp[:, :, ::-1])
    axes[i].axis("off")
    axes[i].set_title(f"Sample idx {idx}")

# 빈 칸 숨기기
for j in range(NUM_SHOW, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.show()
