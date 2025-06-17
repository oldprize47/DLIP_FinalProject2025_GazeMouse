import time
import cv2
import numpy as np
import torch
from PIL import Image
import mediapipe as mp
import pyautogui
from gaze_utils import load_model, get_tf, crop_eyes, LEFT_EYE, RIGHT_EYE, calculate_ear, calibrate, load_calibration, save_calibration, apply_calib_linear, is_inside_circle, draw_face_body_mask


# ────────── 메인 루프 ──────────
def main(calib_file="calib_SH.npy", CKPT="3finetuned_SH.pth", SMOOTH_ALPHA=0.95, FIX_RADIUS=60, FIX_TIME=1.5, UNLOCK_EYE_TIME=1.5, BLINK_COOLTIME=1, CONSEC_FRAMES=2, EAR_THRESHOLD=0.20, CALIB_STD_THRESHOLD=35, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CKPT, device)
    tf = get_tf()
    W, H = pyautogui.size()
    cx, cy = W // 2, H // 2

    # 캘리브레이션 정보 불러오기
    coefs = load_calibration(calib_file)
    calibrated = coefs is not None

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    prev = np.zeros(2, np.float32)
    fix_center = stay_timer = last_free_pos = fixed_since = None
    fixed = False
    blink_count = frame_counter = 0
    last_blink_time = 0
    l_ear = r_ear = 0.3
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    print("[i] c: 새 캘리브레이션  |  space: 저장된 값 사용  |  esc: 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1) & 0xFF
        disp = draw_face_body_mask(frame, alpha=0.68)
        if key == ord("c"):
            coefs = calibrate(cap, model, tf, (W, H), cx, cy, device, std_threshold=CALIB_STD_THRESHOLD)
            save_calibration(calib_file, coefs)
            calibrated = True
            print(">> 캘리브레이션 완료")
            time.sleep(0.4)
            continue
        elif key == ord(" "):
            coefs = load_calibration(calib_file)
            calibrated = coefs is not None
            if calibrated:
                print(">> 기존 캘리브레이션 로드")
            else:
                print("저장된 캘리브레이션 없음")
            time.sleep(0.3)
            continue
        elif key == 27:  # ESC
            break

        if not calibrated or coefs is None:
            cv2.imshow("gaze", disp)
            continue

        patch, lm_dict = crop_eyes(frame, face_mesh)
        if patch is None or patch.size == 0:
            cv2.imshow("gaze", frame)
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
        if not fixed:
            if fix_center is None or not inside:
                fix_center = (mouse_x, mouse_y)
                stay_timer = time.time()
            elif time.time() - stay_timer > FIX_TIME:
                fixed = True
                fixed_since = time.time()
                pyautogui.moveTo(*last_free_pos, _pause=False)

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
        if fixed and last_free_pos:
            pyautogui.moveTo(*last_free_pos, _pause=False)
        else:
            pyautogui.moveTo(gx, gy, _pause=False)

        disp = draw_face_body_mask(frame, alpha=0.68)
        cv2.imshow("gaze", disp)

    cap.release()
    cv2.destroyAllWindows()


# ────────── 파라미터만 바꿔 실행 ──────────
if __name__ == "__main__":
    main(
        calib_file="calib_SH1.npy",  # 캘리브레이션 계수 저장/불러올 파일명 (npy)
        CKPT="3finetuned_SH.pth",  # 모델 파라미터 파일명 (pth)
        SMOOTH_ALPHA=0.95,  # 이전 프레임과 예측값을 섞는 smoothing 계수 (0~1, 클수록 부드럽지만 느림)
        FIX_RADIUS=60,  # 마우스 고정 영역 반지름 (픽셀, 커질수록 더 넓은 영역에서 고정)
        FIX_TIME=1.5,  # 고정되기까지 머무는 시간 (초) (이 시간 이상 같은 자리에 있으면 고정)
        UNLOCK_EYE_TIME=1.5,  # 고정 상태에서 자동 해제되는 시간 (초, 예: 눈 감고 있거나 시간 경과시)
        BLINK_COOLTIME=1,  # 블링크(눈깜빡임) 인식 후 다음 클릭까지 쿨타임 (초)
        CONSEC_FRAMES=2,  # 눈 감은 프레임 연속 개수 (이상일 때만 클릭으로 간주)
        EAR_THRESHOLD=0.20,  # 눈 감음(EAR) 판정 임계값 (낮을수록 예민)
        CALIB_STD_THRESHOLD=35,  # 캘리브레이션 수렴 시 허용 분산(표준편차) 임계값 (작을수록 더 엄격하게 인식)
    )
