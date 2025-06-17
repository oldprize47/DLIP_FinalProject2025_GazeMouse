# File: main.py

import time
import cv2
import numpy as np
import torch
from PIL import Image
import mediapipe as mp
import pyautogui
from gaze_utils import load_model, get_tf, crop_eyes, LEFT_EYE, RIGHT_EYE, calculate_ear, calibrate, load_calibration, save_calibration, apply_calib_linear, is_inside_circle, draw_face_body_mask

# ----------------- Global Config -----------------

CALIB_FILE = "calib_SH.npy"  # Calibration file path
CKPT = "finetuned_SH.pth"  # Model checkpoint path
SMOOTH_ALPHA = 0.95  # Smoothing factor for gaze output (0~1, higher = smoother)
FIX_RADIUS = 60  # Radius (pixels) for fixation detection
FIX_TIME = 1.5  # Time (sec) required to fixate before locking
UNLOCK_EYE_TIME = 1.5  # Time (sec) after fixation before unlocking by blink
BLINK_COOLTIME = 1  # Minimum interval (sec) between allowed blinks/clicks
CONSEC_FRAMES = 2  # Number of consecutive frames to confirm a blink
EAR_THRESHOLD = 0.20  # Eye Aspect Ratio threshold for blink detection
CALIB_STD_THRESHOLD = 35  # Standard deviation threshold for calibration stability
WIN_NAME = "gaze"  # OpenCV window name

# --------- UI size config (adapts to screen ratio) ----------
W, H = pyautogui.size()  # Screen width, height
ASPECT = W / H  # Screen aspect ratio (width/height)
CENTER_BASE = 800  # Base size for main window (height)
SMALL_BASE = 80  # Base size for small window (height)
CENTER_SIZE = (int(CENTER_BASE * ASPECT), int(CENTER_BASE))  # Main window size (w, h)
SMALL_SIZE = (int(SMALL_BASE * ASPECT), int(SMALL_BASE))  # Small window size (w, h)
SMALL_X = 0  # X position for small window (left edge)
MARGIN = 60  # Margin from bottom edge for small window
SMALL_Y = max(0, H - SMALL_SIZE[1] - MARGIN)  # Y position for small window (bottom, with margin)
CENTER_X = W // 2 - CENTER_SIZE[0] // 2  # Main window X (centered)
CENTER_Y = H // 2 - CENTER_SIZE[1] // 2  # Main window Y (centered)
# --------------------------------------------------


def main(calib_file=CALIB_FILE, CKPT=CKPT, SMOOTH_ALPHA=SMOOTH_ALPHA, FIX_RADIUS=FIX_RADIUS, FIX_TIME=FIX_TIME, UNLOCK_EYE_TIME=UNLOCK_EYE_TIME, BLINK_COOLTIME=BLINK_COOLTIME, CONSEC_FRAMES=CONSEC_FRAMES, EAR_THRESHOLD=EAR_THRESHOLD, CALIB_STD_THRESHOLD=CALIB_STD_THRESHOLD, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CKPT, device)
    tf = get_tf()
    cx, cy = W // 2, H // 2
    coefs = load_calibration(calib_file)
    calibrated = coefs is not None

    show_small = False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, *CENTER_SIZE)
    cv2.moveWindow(WIN_NAME, CENTER_X, CENTER_Y)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 1)

    prev = np.zeros(2, np.float32)
    fix_center = stay_timer = last_free_pos = fixed_since = None
    fixed = False
    blink_count = frame_counter = 0
    last_blink_time = 0
    l_ear = r_ear = 0.3
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    print("\n\n[i] c: Calibrate | space: Load calibration | m: Move window | esc: Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        disp = cv2.resize(frame, CENTER_SIZE)
        key = cv2.waitKey(1) & 0xFF

        # -------- Hotkeys --------
        if key == ord("c"):
            # Start calibration
            coefs = calibrate(cap, model, tf, (W, H), cx, cy, device, face_mesh, std_threshold=CALIB_STD_THRESHOLD)
            save_calibration(calib_file, coefs)
            calibrated = True
            print(">> Calibration completed")
            time.sleep(0.4)
            continue
        elif key == ord(" "):
            # Load calibration file
            coefs = load_calibration(calib_file)
            calibrated = coefs is not None
            print(">> Calibration loaded" if calibrated else "No calibration found")
            time.sleep(0.3)
            continue
        elif key == ord("m"):
            # Toggle between main/small display window
            show_small = not show_small
        elif key == 27:  # ESC to quit
            break

        if not calibrated or coefs is None:
            cv2.imshow(WIN_NAME, disp)
            continue

        patch, lm_dict = crop_eyes(frame, face_mesh)
        if patch is None or patch.size == 0:
            cv2.imshow(WIN_NAME, frame)
            continue

        pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            pred = model(tf(pil).unsqueeze(0).to(device)).cpu().squeeze().numpy()
        smoothed = SMOOTH_ALPHA * prev + (1 - SMOOTH_ALPHA) * pred
        prev[:] = smoothed
        gaze_xy = apply_calib_linear(smoothed, coefs, cx, cy)
        gx = int(np.clip(gaze_xy[0], 5, W - 5))
        gy = int(np.clip(gaze_xy[1], 5, H - 5))
        mouse_x, mouse_y = pyautogui.position()
        last_free_pos = (mouse_x, mouse_y)
        inside = is_inside_circle(fix_center, (mouse_x, mouse_y), FIX_RADIUS) if fix_center else False

        # ------------- Fixation Logic -------------
        if not fixed:
            if fix_center is None or not inside:
                fix_center = (mouse_x, mouse_y)
                stay_timer = time.time()
            elif time.time() - stay_timer > FIX_TIME:
                fixed = True
                fixed_since = time.time()
                pyautogui.moveTo(*last_free_pos, _pause=False)

        # ------------- Blink Detection for Click -------------
        if lm_dict and all(i in lm_dict for i in LEFT_EYE + RIGHT_EYE):
            l_ear = calculate_ear(LEFT_EYE, lm_dict)
            r_ear = calculate_ear(RIGHT_EYE, lm_dict)
        if fixed:
            both_eyes_closed = l_ear < EAR_THRESHOLD and r_ear < EAR_THRESHOLD
            if time.time() - fixed_since > UNLOCK_EYE_TIME:
                fixed = False
                fix_center = None
                fixed_since = None
                continue
            if both_eyes_closed:
                frame_counter += 1
            else:
                frame_counter = 0
            if frame_counter >= CONSEC_FRAMES and time.time() - last_blink_time > BLINK_COOLTIME:
                pyautogui.click()
                blink_count += 1
                last_blink_time = time.time()
                fixed = False
                fix_center = None
                frame_counter = 0
        # ----------- Mouse Movement -----------
        if fixed and last_free_pos:
            pyautogui.moveTo(*last_free_pos, _pause=False)
        else:
            pyautogui.moveTo(gx, gy, _pause=False)

        disp = draw_face_body_mask(frame, alpha=0.68)
        # Handle window resize/move for small or main mode
        if show_small:
            cv2.resizeWindow(WIN_NAME, *SMALL_SIZE)
            cv2.moveWindow(WIN_NAME, SMALL_X, SMALL_Y)
            disp_to_show = cv2.resize(disp, SMALL_SIZE)
        else:
            cv2.resizeWindow(WIN_NAME, *CENTER_SIZE)
            cv2.moveWindow(WIN_NAME, CENTER_X, CENTER_Y)
            disp_to_show = cv2.resize(disp, CENTER_SIZE)
        cv2.imshow(WIN_NAME, disp_to_show)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
