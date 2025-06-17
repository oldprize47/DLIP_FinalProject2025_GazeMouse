# File: data_utils.py

import os
import cv2
import numpy as np
import pandas as pd


def make_targets(N, W, H):
    """
    Generate N target (x, y) positions spread evenly across a W x H area.
    Returns a list of (x, y) tuples in snake order (zigzag columns).
    """
    aspect = W / H
    rows = int(round((N / aspect) ** 0.5))
    cols = int(round(aspect * rows))
    x_list = np.linspace(0, W - 1, cols, dtype=int)
    y_list = np.linspace(0, H - 1, rows, dtype=int)
    targets = []
    for idx_col, x in enumerate(x_list):
        # Create column with alternating direction for snake pattern
        col_targets = [(x, y) for y in (y_list if idx_col % 2 == 0 else y_list[::-1])]
        if idx_col % 2 == 1:
            col_targets = col_targets[::-1]
        targets.extend(col_targets)
    return targets


def show_message_on_bg(bg, message, duration=1000):
    """
    Display a message on the background image for a certain duration (ms).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    color = (0, 0, 255)  # Red text
    thickness = 4
    text_size, _ = cv2.getTextSize(message, font, font_scale, thickness)
    text_x = (bg.shape[1] - text_size[0]) // 2
    text_y = (bg.shape[0] + text_size[1]) // 2 + 100
    bg_copy = bg.copy()
    cv2.putText(bg_copy, message, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow("calib_point", bg_copy)
    cv2.waitKey(duration)


def save_csv(patches, labels, csv_path):
    """
    Save image paths and corresponding (dx, dy) labels to CSV file.
    """
    df = pd.DataFrame({"image_path": patches, "dx": [l[0] for l in labels], "dy": [l[1] for l in labels]})
    df.to_csv(csv_path, index=False)


def load_csv(csv_path):
    """
    Load image paths and labels from a CSV file.
    If not found, return empty lists.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        patches = df["image_path"].tolist()
        labels = list(zip(df["dx"], df["dy"]))
        print(f"Loaded {len(patches)} patches from existing CSV")
        return patches, labels
    else:
        return [], []
