# File: make_csv_custom.py

import os
import cv2
import numpy as np
import pyautogui
import random
from PIL import Image
import mediapipe as mp
from data_utils import make_targets, show_message_on_bg, save_csv, load_csv
from gaze_utils import crop_eyes

# ========== [1] Configuration ==========

SAVE_DIR = "p01"  # Folder to save captured images
CSV_PATH = "p01.csv"  # Output label CSV file
N = 200  # Total number of target points
PATCH_PER_POINT = 10  # Number of images to save per point
ORDER_FILE = "targets_order.npy"  # File to save random order of targets

# ========================================


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    W, H = pyautogui.size()
    targets = make_targets(N, W, H)
    print(f"Grid computed: cols, rows => actual number of points: {len(targets)}")
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    patches, labels = load_csv(CSV_PATH)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # ===== [Random order & Resume management] =====
    if os.path.exists(ORDER_FILE):
        targets_order = np.load(ORDER_FILE, allow_pickle=True)
        print(f"[Resume] Loaded random order ({len(targets_order)} points)")
        # Check for changed grid size/resolution
        if len(targets_order) != len(targets):
            print(f"[Warning] Current grid ({len(targets)}) and saved order ({len(targets_order)}) are different.")
            print("Complete your previous session before changing N/grid/settings.")
            exit(1)
        targets = targets_order.tolist()
    else:
        random.shuffle(targets)
        np.save(ORDER_FILE, np.array(targets, dtype=object))
        print(f"[New] Random order file created ({len(targets)} points)")

    idx = len(patches) // PATCH_PER_POINT  # Resume support
    cx, cy = W // 2 - 100, H // 2

    while idx < len(targets):
        tx, ty = targets[idx]
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        progress_text = f"[{idx+1} / {len(targets)}]"
        cv2.putText(bg, progress_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.circle(bg, (tx, ty), 30, (0, 255, 0), -1)
        line_len = 40
        cv2.line(bg, (tx - line_len, ty), (tx + line_len, ty), (0, 0, 0), 5)
        cv2.line(bg, (tx, ty - line_len), (tx, ty + line_len), (0, 0, 0), 5)
        cv2.namedWindow("calib_point", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("calib_point", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        go_back = False

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            patch = crop_eyes(frame, face_mesh, return_landmarks=False)
            cv2.imshow("calib_point", bg)
            if patch is not None:
                patch_show = cv2.resize(patch, (224, 224))
                cv2.imshow("patch", patch_show)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                print("Exiting and saving all work...")
                cap.release()
                cv2.destroyAllWindows()
                save_csv(patches, labels, CSV_PATH)
                print("CSV and images saved!")
                return
            elif key == ord(" "):  # Space: Save PATCH_PER_POINT images at this location
                tmp_patches = []
                tmp_labels = []
                success = True
                for i in range(PATCH_PER_POINT):
                    ret2, frame2 = cap.read()
                    if not ret2:
                        success = False
                        break
                    frame2 = cv2.flip(frame2, 1)
                    patch2 = crop_eyes(frame2, face_mesh, return_landmarks=False)
                    if patch2 is not None:
                        patch_pil2 = Image.fromarray(cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB))
                        out_path2 = os.path.join(SAVE_DIR, f"{idx:03d}_{len(patches)+len(tmp_patches):04d}.jpg")
                        patch_pil2.save(out_path2)
                        dx = tx - W // 2
                        dy = ty - H // 2
                        tmp_patches.append(out_path2)
                        tmp_labels.append([dx, dy])
                    else:
                        # Failure: Remove all saved so far
                        for path in tmp_patches:
                            if os.path.exists(path):
                                os.remove(path)
                        show_message_on_bg(bg, "Failed! Eye not detected\nTry again", 700)
                        success = False
                        break
                    cv2.waitKey(80)  # Short delay for consecutive capture
                if success and len(tmp_patches) == PATCH_PER_POINT:
                    patches.extend(tmp_patches)
                    labels.extend(tmp_labels)
                    show_message_on_bg(bg, f"Saved {PATCH_PER_POINT} images!", 100)
                    break  # Proceed to next point
            elif key == ord("z"):
                # Cancel/correct last PATCH_PER_POINT images
                num_to_remove = min(PATCH_PER_POINT, len(patches))
                if num_to_remove > 0:
                    for _ in range(num_to_remove):
                        del_img = patches.pop()
                        labels.pop()
                        if os.path.exists(del_img):
                            os.remove(del_img)
                    if idx > 0:
                        idx -= 1
                        show_message_on_bg(bg, "Canceled!", 1000)
                        go_back = True
                        break
                else:
                    show_message_on_bg(bg, "No capture to cancel!", 1000)
        cv2.destroyWindow("patch")
        if go_back:
            continue
        idx += 1

    cap.release()
    cv2.destroyAllWindows()
    save_csv(patches, labels, CSV_PATH)
    print("CSV and images saved!")

    # ===== [Delete ORDER file on complete] =====
    if os.path.exists(ORDER_FILE):
        os.remove(ORDER_FILE)
        print(f"All grid points complete! {ORDER_FILE} deleted.")


# ========== [4] Run ==========

if __name__ == "__main__":
    main()
