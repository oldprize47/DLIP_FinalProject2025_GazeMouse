# fginet_eyes.py
import torch
import torch.nn as nn
from fginet import FGINet  # 기존 백본


class FGINetEyes(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FGINet()  # (face → 4값: head_p, head_y, gaze_p, gaze_y)

        # 눈 전용 CNN
        self.eye_conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56×56
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28×28
            nn.Flatten(),  # → eye_feat_dim
        )
        eye_feat_dim = 64 * 28 * 28

        # 눈 특징만으로 gaze (yaw, pitch) 예측
        self.fc_gaze = nn.Sequential(
            nn.Linear(eye_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, face, eyes):
        # 1) 백본으로 머리( yaw, pitch )만 얻기
        backbone_out = self.backbone(face)  # (B,4)
        head_out = backbone_out[:, :2]  # (B,2)

        # 2) 눈 특징으로 gaze 예측
        eyes_feat = self.eye_conv(eyes)  # (B, eye_feat_dim)
        gaze_out = self.fc_gaze(eyes_feat)  # (B,2)

        # 3) concat → (B,4): [head_yaw, head_pitch, gaze_yaw, gaze_pitch]
        out = torch.cat([head_out, gaze_out], dim=1)
        return out
