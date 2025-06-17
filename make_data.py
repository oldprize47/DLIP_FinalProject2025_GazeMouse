# 파일명: make_data.py
import os
import cv2
import numpy as np
import pyautogui
import random
from PIL import Image
import mediapipe as mp
from data_utils import make_targets, show_message_on_bg, save_csv, load_csv
from gaze_utils import crop_eyes

# ========== [1] 설정값 ==========
SAVE_DIR = "p01"  # 이미지 저장 폴더
CSV_PATH = "p01.csv"  # 라벨 CSV
N = 200  # 총 타깃 점 개수
PATCH_PER_POINT = 10  # 한 점에서 저장할 패치 개수
ORDER_FILE = "targets_order.npy"  # 랜덤 순서 저장 파일


# ========== [2] main ==========
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    W, H = pyautogui.size()
    targets = make_targets(N, W, H)
    print(f"자동 계산: cols, rows => 실제 점 개수: {len(targets)}")
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    patches, labels = load_csv(CSV_PATH)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # ===== [랜덤 순서/이어하기 관리] =====
    if os.path.exists(ORDER_FILE):
        targets_order = np.load(ORDER_FILE, allow_pickle=True)
        print(f"[이어하기] 랜덤 순서 로드 ({len(targets_order)}개)")
        # 점 개수/해상도 변경 체크
        if len(targets_order) != len(targets):
            print(f"[경고] 현재 그리드 개수({len(targets)})와 저장된 랜덤 순서({len(targets_order)})가 다릅니다.")
            print("그리드/점 개수/N 변경 전, 기존 session(데이터)을 먼저 완주하세요.")
            exit(1)
        targets = targets_order.tolist()
    else:
        random.shuffle(targets)
        np.save(ORDER_FILE, np.array(targets, dtype=object))
        print(f"[신규] 랜덤 순서 파일 생성 ({len(targets)}개)")

    idx = len(patches) // PATCH_PER_POINT  # 이어하기 지원
    cx, cy = W // 2 - 100, H // 2
    while idx < len(targets):
        tx, ty = targets[idx]
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        progress_text = f"[{idx+1} / {len(targets)}]"
        cv2.putText(bg, progress_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5, cv2.LINE_AA)  # 좌상단 (x, y) 위치 (원하면 조절)  # 폰트 크기  # 흰색  # 두께
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
            if key == 27:  # ESC 전체 종료
                print("작업을 종료합니다.")
                cap.release()
                cv2.destroyAllWindows()
                save_csv(patches, labels, CSV_PATH)
                print("CSV 및 패치 저장 완료!")
                return
            elif key == ord(" "):  # 스페이스: 이미지 10장 연속 저장
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
                        # 실패: 지금까지 저장한 것도 다 삭제
                        for path in tmp_patches:
                            if os.path.exists(path):
                                os.remove(path)
                        show_message_on_bg(bg, f"실패! 눈 인식 불가\n다시 시도하세요", 700)
                        success = False
                        break
                    cv2.waitKey(80)  # 짧은 딜레이로 연속 캡처
                if success and len(tmp_patches) == PATCH_PER_POINT:
                    patches.extend(tmp_patches)
                    labels.extend(tmp_labels)
                    show_message_on_bg(bg, f"Saved {PATCH_PER_POINT} images!", 100)
                    break  # 다음 점으로 넘어감
            elif key == ord("z"):
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
    print("CSV 및 패치 저장 완료!")

    # ===== [완료 시: ORDER 파일 삭제] =====
    if os.path.exists(ORDER_FILE):
        os.remove(ORDER_FILE)
        print(f"모든 그리드 완료! {ORDER_FILE} 파일 삭제됨.")


# ========== [4] 실행 ==========
if __name__ == "__main__":
    main()
