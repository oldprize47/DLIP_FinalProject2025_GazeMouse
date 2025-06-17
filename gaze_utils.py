# File: gaze_utils.py

import os, time, random
import numpy as np
from math import hypot
from PIL import Image
import cv2
import torch

# -------------------- Model Loading, Transform, etc. --------------------


def load_model(ckpt_path, device):
    """
    Load FGINet model from checkpoint.
    """
    from fginet import FGINet

    model = FGINet().to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    return model


def get_tf():
    """
    Return image transform for inference.
    """
    from eye_patch_dataset import get_infer_transform

    return get_infer_transform()


# -------------------- Eye Patch Extraction --------------------


def crop_eyes(frame, face_mesh, img_size=224, margin=0.6, return_landmarks=True):
    """
    Extract eye patch and landmarks from the frame using MediaPipe FaceMesh.
    Returns padded eye patch (img_size x img_size), and facial landmarks.
    """
    LEFT = [33, 133, 160, 159, 158, 157, 173, 246]
    RIGHT = [362, 263, 387, 386, 385, 384, 398, 466]
    h, w = frame.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return (None, {}) if return_landmarks else None

    lm = res.multi_face_landmarks[0].landmark
    xs = [lm[i].x for i in LEFT + RIGHT]
    ys = [lm[i].y for i in LEFT + RIGHT]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx_lm, cy_lm = (xmin + xmax) / 2, (ymin + ymax) / 2
    w_box = (xmax - xmin) * (margin + 1)
    h_box = (ymax - ymin) * (margin + 4.5)
    xmin, xmax = cx_lm - w_box / 2, cx_lm + w_box / 2
    ymin, ymax = cy_lm - h_box / 2, cy_lm + h_box / 2
    x1, x2 = int(max(0, xmin * w)), int(min(w - 1, xmax * w))
    y1, y2 = int(max(0, ymin * h)), int(min(h - 1, ymax * h))
    eye_patch = frame[y1:y2, x1:x2]
    if eye_patch.size == 0:
        return (None, {}) if return_landmarks else None
    ph, pw = eye_patch.shape[:2]
    scale = img_size / max(ph, pw)
    resized = cv2.resize(eye_patch, (int(pw * scale), int(ph * scale)))
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    y_off = (img_size - resized.shape[0]) // 2
    x_off = (img_size - resized.shape[1]) // 2
    canvas[y_off : y_off + resized.shape[0], x_off : x_off + resized.shape[1]] = resized
    lm_dict = {idx: (int(pt.x * w), int(pt.y * h)) for idx, pt in enumerate(lm)}
    return (canvas, lm_dict) if return_landmarks else canvas


# -------------------- EAR-based Blink Detection --------------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def calculate_ear(eye_points, landmarks):
    """
    Compute Eye Aspect Ratio (EAR) from eye landmarks.
    """
    p1, p2, p3, p4, p5, p6 = [landmarks[idx] for idx in eye_points]
    ear = (hypot(p2[0] - p6[0], p2[1] - p6[1]) + hypot(p3[0] - p5[0], p3[1] - p5[1])) / (2.0 * hypot(p1[0] - p4[0], p1[1] - p4[1]))
    return ear


# -------------------- Calibration (Least Squares) --------------------
def show_calib_intro(W, H):
    """Show calibration instruction fullscreen until key pressed."""
    bg = np.full((H, W, 3), (70, 70, 70), dtype=np.uint8)
    msg1 = "Calibration Mode"
    msg2 = "Look at the RED dot when it appears."
    msg3 = "Keep gazing for ~1.5 sec until it turns GREEN."
    msg4 = "(Press SPACE to start calibration...)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.7
    color = (255, 255, 255)
    thickness = 3

    cv2.putText(bg, msg1, (W // 2 - 330, H // 2 - 80), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(bg, msg2, (W // 2 - 330, H // 2), font, 1.2, color, 2, cv2.LINE_AA)
    cv2.putText(bg, msg3, (W // 2 - 330, H // 2 + 50), font, 1.2, color, 2, cv2.LINE_AA)
    cv2.putText(bg, msg4, (W // 2 - 330, H // 2 + 140), font, 1.0, (200, 220, 255), 2, cv2.LINE_AA)

    cv2.namedWindow("calib_full", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("calib_full", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        cv2.imshow("calib_full", bg)
        key = cv2.waitKey(20) & 0xFF
        if key in [ord(" "), 13, 10]:  # space, enter
            break
        elif key == 27:
            cv2.destroyWindow("calib_full")
            exit(0)


def calibrate(cap, model, tf, screen_size, cx, cy, device, face_mesh, min_time=2, required_stable=20, std_threshold=35):
    """
    Calibrate using gaze predictions and screen targets.
    Returns linear regression coefficients for mapping prediction to screen.
    """
    W, H = screen_size
    preds, targets = [], []
    cv2.namedWindow("calib_full", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("calib_full", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    n_x, n_y = 5, 3
    margin_x, margin_y = 0.03, 0.03
    x_ratios = np.linspace(0 + margin_x, 1 - margin_x, n_x)
    y_ratios = np.linspace(0 + margin_y, 1 - margin_y, n_y)
    calib_targets = [(int(round(x * (W - 1))), int(round(y * (H - 1)))) for y in y_ratios for x in x_ratios]

    # Add corner, center, and edge points for more robust calibration
    corners = [(5, 5), (W - 6, 5), (5, H - 6), (W - 6, H - 6)]
    center = [(W // 2, H // 2)]
    middles = [(W // 2, 5), (W // 2, H - 6), (5, H // 2), (W - 6, H // 2)]
    bonus_points = corners + center + middles
    calib_targets = list({(int(x), int(y)) for (x, y) in calib_targets + bonus_points})
    random.shuffle(calib_targets)
    GRAY = (100, 100, 100)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    show_calib_intro(W, H)
    for idx, (tx, ty) in enumerate(calib_targets):
        bg = np.full((H, W, 3), GRAY, dtype=np.uint8)
        cv2.circle(bg, (tx, ty), 30, (0, 0, 255), -1)  # Red dot
        line_len = 30
        cv2.line(bg, (tx - line_len, ty), (tx + line_len, ty), BLACK, 5)
        cv2.line(bg, (tx, ty - line_len), (tx, ty + line_len), BLACK, 5)
        cv2.circle(bg, (tx, ty), 4, WHITE, -1)
        buf = []
        t_start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            patch, _ = crop_eyes(frame, face_mesh)
            if patch is None:
                cv2.imshow("calib_full", bg)
                cv2.waitKey(1)
                continue
            pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            inp = tf(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(inp).cpu().squeeze().numpy()
            buf.append(pred)
            if len(buf) > required_stable:
                buf = buf[-required_stable:]

            now = time.time()
            if (now - t_start > min_time) and len(buf) == required_stable:
                arr = np.array(buf)
                std = arr.std(axis=0)
                if np.all(std < std_threshold):
                    bg2 = bg.copy()
                    cv2.circle(bg2, (tx, ty), 30, (0, 255, 0), -1)  # Green dot if stable
                    cv2.imshow("calib_full", bg2)
                    cv2.waitKey(400)
                    break

            cv2.imshow("calib_full", bg)
            if cv2.waitKey(1) & 0xFF == 27:
                print("Calibration canceled!")
                exit(0)
        mean_pred = np.mean(buf, axis=0)
        preds.append(mean_pred)
        targets.append([tx - cx, ty - cy])
    cv2.destroyWindow("calib_full")

    # Linear regression (least squares)
    preds = np.array(preds)
    targets = np.array(targets)

    def linear_features(arr):
        x, y = arr[:, 0], arr[:, 1]
        return np.stack([np.ones_like(x), x, y], axis=1)  # (N, 3)

    X_lin = linear_features(preds)
    coef_x, _, _, _ = np.linalg.lstsq(X_lin, targets[:, 0], rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(X_lin, targets[:, 1], rcond=None)
    return (coef_x, coef_y)


# -------------------- Calibration File I/O --------------------


def load_calibration(calib_file):
    """
    Load calibration coefficients from file.
    """
    if os.path.exists(calib_file):
        return np.load(calib_file, allow_pickle=True)
    return None


def save_calibration(calib_file, coefs):
    """
    Save calibration coefficients to file.
    """
    np.save(calib_file, coefs)


# -------------------- Prediction to Screen Coordinate --------------------


def apply_calib_linear(pred, coefs, cx, cy):
    """
    Map predicted (dx, dy) to screen coordinates using calibration.
    """
    dx, dy = pred[0], pred[1]
    feat = np.array([1.0, dx, dy])
    tx = np.dot(feat, coefs[0])
    ty = np.dot(feat, coefs[1])
    screen_xy = np.array([cx, cy]) + np.array([tx, ty])
    return screen_xy


# -------------------- Fixation Region Check --------------------


def is_inside_circle(center, point, r):
    """
    Check if point is inside circle centered at 'center' with radius r.
    """
    return (center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2 <= r**2


# -------------------- Visualization --------------------


def draw_face_body_mask(img, alpha=0.7, face_radius_ratio=0.34, body_width_ratio=0.4, body_height_ratio=0.3):
    """
    Overlay transparent mask for face and body regions.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    face_center = (w // 2, int(h * 0.5))
    face_radius = int(min(w, h) * face_radius_ratio)
    cv2.circle(mask, face_center, face_radius, 255, -1)
    body_center = (w // 2, int(h))
    body_axes = (int(w * body_width_ratio), int(h * body_height_ratio))
    cv2.ellipse(mask, body_center, body_axes, 0, 0, 360, 255, -1)
    mask_inv = cv2.bitwise_not(mask)
    colored = np.zeros_like(img)
    overlay = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)
    result = img.copy()
    result[mask_inv > 0] = overlay[mask_inv > 0]
    return result
