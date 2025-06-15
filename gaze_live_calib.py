import cv2, torch, numpy as np, pyautogui, time
import mediapipe as mp
from PIL import Image
from torchvision import transforms
from eye_patch_dataset import EyePatchDataset, get_infer_transform
from fginet import FGINet
import time

# ========== Mediapipe 준비 ==========
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
LEFT = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT = [362, 263, 387, 386, 385, 384, 398, 466]

# ========== 환경 ==========
CKPT = "finetuned_SH.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOOTH_ALPHA = 0.9
model = FGINet().to(DEVICE).eval()
ckpt = torch.load(CKPT, map_location=DEVICE)
if isinstance(ckpt, dict) and "model" in ckpt:
    model.load_state_dict(ckpt["model"])
else:
    model.load_state_dict(ckpt)
tf = get_infer_transform()
W, H = pyautogui.size()
print(f"W: {W}, H: {H}")
cx, cy = W // 2, H // 2


def crop_eyes(frame, face_mesh=face, img_size=224, margin=0.6):
    h, w = frame.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    xs = [lm[i].x for i in LEFT + RIGHT]
    ys = [lm[i].y for i in LEFT + RIGHT]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    w_box = (xmax - xmin) * (margin + 1)
    h_box = (ymax - ymin) * (margin + 4.5)
    xmin, xmax = cx - w_box / 2, cx + w_box / 2
    ymin, ymax = cy - h_box / 2, cy + h_box / 2
    x1, x2 = int(max(0, xmin * w)), int(min(w - 1, xmax * w))
    y1, y2 = int(max(0, ymin * h)), int(min(h - 1, ymax * h))
    eye_patch = frame[y1:y2, x1:x2]
    patch_h, patch_w = eye_patch.shape[:2]
    scale = img_size / max(patch_h, patch_w)
    resized = cv2.resize(eye_patch, (int(patch_w * scale), int(patch_h * scale)))
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    y_off = (img_size - resized.shape[0]) // 2
    x_off = (img_size - resized.shape[1]) // 2
    canvas[y_off : y_off + resized.shape[0], x_off : x_off + resized.shape[1]] = resized
    return canvas


# ========== 캘리브레이션 지점 ==========
x_ratios = [0.1, 0.5, 0.9]
y_ratios = [0.1, 0.5, 0.9]
calib_targets = [(int(W * x), int(H * y)) for y in y_ratios for x in x_ratios]


# ========== 캘리브레이션 함수 ==========
def calibrate(cap, min_time=1.5, required_stable=18, std_threshold=30):
    preds, targets = [], []
    cv2.namedWindow("calib_full", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("calib_full", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("\n===== Calibration Mode =====")
    print("각 점에 '빨간 점'이 나오면 그 점을 바라봐 주세요!")
    print("잠시(1.5초~)동안 충분히 응시하면 다음 점으로 넘어갑니다.\n")
    for idx, (tx, ty) in enumerate(calib_targets):
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.circle(bg, (tx, ty), 30, (0, 0, 255), -1)
        buf = []
        t_start = time.time()
        last_print = time.time()
        print(f"[{idx+1}/{len(calib_targets)}] {tx},{ty} 위치를 바라보세요 (점이 초록색으로 바뀌면 성공!)")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            patch = crop_eyes(frame)
            if patch is None:
                cv2.imshow("calib_full", bg)
                cv2.waitKey(1)
                continue
            pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            inp = tf(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(inp).cpu().squeeze().numpy()
            buf.append(pred)
            if len(buf) > required_stable:
                buf = buf[-required_stable:]  # 최신 required_stable개만 유지

            now = time.time()
            # **1초마다 프린트**
            if now - last_print > 1.0 and len(buf) > 3:
                arr = np.array(buf)
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                print(f"  예측값 분포 (최근 {len(buf)}개):")
                print(f"    X = {arr[:,0].min():.1f} ~ {arr[:,0].max():.1f},  평균 {mean[0]:.1f}, std {std[0]:.2f}")
                print(f"    Y = {arr[:,1].min():.1f} ~ {arr[:,1].max():.1f},  평균 {mean[1]:.1f}, std {std[1]:.2f}")
                last_print = now

            if (now - t_start > min_time) and len(buf) == required_stable:
                arr = np.array(buf)
                std = arr.std(axis=0)
                if np.all(std < std_threshold):
                    bg2 = bg.copy()
                    cv2.circle(bg2, (tx, ty), 30, (0, 255, 0), -1)
                    cv2.imshow("calib_full", bg2)
                    cv2.waitKey(400)
                    break

            cv2.imshow("calib_full", bg)
            cv2.imshow("patch", patch)
            if cv2.waitKey(1) & 0xFF == 27:
                print("캘리브레이션 취소!")
                exit(0)
        mean_pred = np.mean(buf, axis=0)
        preds.append(mean_pred)
        targets.append([tx - cx, ty - cy])
    cv2.destroyWindow("calib_full")
    cv2.destroyWindow("patch")
    # 최소제곱 보정
    preds = np.array(preds)
    targets = np.array(targets)
    ones = np.ones((preds.shape[0], 1))
    X = np.hstack([preds, ones])
    A, _, _, _ = np.linalg.lstsq(X, targets, rcond=None)
    return A


def apply_calib(pred, A):
    # pred: (2,), A: (3,2)
    x = np.append(pred, 1.0)  # [dx, dy, 1]
    offset = np.dot(x, A)  # (2,) [tx-cx, ty-cy]
    screen_xy = np.array([cx, cy]) + offset
    return screen_xy


# ========== 실시간 루프 ==========
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    prev = np.array([0.0, 0.0], dtype=np.float32)
    calibrated = False
    A = None
    print("[i] c 키를 누르면 캘리브레이션 시작")
    print("[i] esc 키로 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1)
        if key == ord("c") and not calibrated:
            # --- 캘리브레이션 ---
            A = calibrate(cap)
            calibrated = True
            print("캘리브레이션 완료! 실시간 추적 시작")
            time.sleep(0.5)
            continue
        if key == 27:  # ESC
            break

        patch = crop_eyes(frame)
        if patch is None or not calibrated:
            cv2.imshow("gaze", frame)
            continue
        pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        inp = tf(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(inp).cpu().squeeze().numpy()
        smoothed = SMOOTH_ALPHA * prev + (1 - SMOOTH_ALPHA) * pred
        prev = smoothed

        # 보정 적용
        screen_xy = apply_calib(smoothed, A)
        x_px, y_px = int(screen_xy[0]), int(screen_xy[1])
        pyautogui.moveTo(x_px, y_px, _pause=False)

        # 디버그 표시
        draw_x = int(frame.shape[1] // 2 + smoothed[0] * (frame.shape[1] / W))
        draw_y = int(frame.shape[0] // 2 + smoothed[1] * (frame.shape[0] / H))
        cv2.circle(frame, (draw_x, draw_y), 6, (0, 255, 0), -1)
        cv2.imshow("gaze", frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
