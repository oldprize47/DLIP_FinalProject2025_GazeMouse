# 파일명: fit_calib.py  (바이어스 저장)
import json
import numpy as np

data = json.load(open("calib_center.json", "r"))
gy0 = data["gaze_yaw0"]
gp0 = data["gaze_pitch0"]

# 저장 (bias 만)
np.savez("gaze_bias.npz", gy0=gy0, gp0=gp0)
print("gaze_bias.npz saved:", data)
