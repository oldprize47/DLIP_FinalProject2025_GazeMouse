# [DLIP 2025-1] Gaze Tracking Mouse



## AI Gaze Mouse with eye gaze

------

**Date:** 2025-06-20

**Author:** Sangheon Park(21800275), Sunwoo Kim(22000090)

**Github:** [DLIP_FinalProject2025_GazeMouse](https://github.com/oldprize47/DLIP_FinalProject2025_GazeMouse.git)

**Demo Video:** [[DLIP 2025-1] Gaze Tracking Mouse](https://www.youtube.com/watch?v=VR9T6X-zanU)

------

### 1. Introduction

​	In this study, we highlight the need for gaze estimation technology capable of controlling user interfaces using only a laptop’s built-in camera—without any additional equipment—even in hands-free environments, amid the rapid expansion of the VR/AR industry. Unlike conventional methods that depend on expensive, high-resolution hardware, we propose and implement FGI-Net (Fusion Global Information Network), a deep learning model that achieves sufficient accuracy by leveraging standard laptop camera footage as input.

**Goal** 

​	Develop FGI-Net, a deep learning model that achieves high-accuracy gaze estimation using only a laptop’s built-in camera, and implement a real-time gaze-controlled mouse with blink-based clicking.

**Specific Goal**

- Reliable gaze detection with a standard webcam 

- Precise cursor positioning 

- Accurate click implementation 

------

### 2. Requirement

#### Hardware

- **CPU**: Intel Core i9-13980HX @ 2.20 GHz  
- **GPU**: NVIDIA GeForce RTX 4080 (CUDA 12.1 support)  
- **RAM**: 64 GB DDR5 @ 5600 MHz (2 × 32 GB Micron modules)  
- **Webcam**: Integrated Camera – 1280 × 720 @ 30 fps 

#### Software Installation

- **Python**: 3.11.12

- **PyTorch**: 2.5.1 (CUDA 12.1)

- **OpenCV (opencv-python)**: 4.11.0.86

- **NumPy**: 2.2.6

- **Pandas**: 2.2.3  

- **SciPy**: 1.15.2  

  > For the full list of additional dependencies, please refer to the `environment.yml` file on GitHub.

------

### 3. Tutorial Procedure

#### Flow Chart



#### Setting Environment

**Anaconda Install**

[Anaconda - DLIP (gitbook.io)](https://ykkim.gitbook.io/dlip/installation-guide/anaconda)

**Download Files**

[DLIP_FinalProject2025_GazeMouse](https://github.com/oldprize47/DLIP_FinalProject2025_GazeMouse.git)

**Create virtual environment**

Run anaconda prompt in administrator mode.

```cmd
# 1) From the project root, create the env from the yml file
conda env create -f environment.yml

# 2) Activate the environment
conda activate Gaze_mouse_fgi
```

#### Code Desription

- **fginet.py:** The model used for training
- **make_csv_custom.py:** Script for creating custom datasets
- **make_csv_mpiigaze.py:** Script for generating CSV if using the MPIIGaze dataset
- **data_utils.py:** Helper functions for data preparation
- **train.py:** Code for training the model
- **train_utils.py:** Utility functions used during training
- **eye_patch_dataset.py:** Image resizing and dataset class for training
- **main.py:** Script to run the gaze-estimation mouse
- **gaze_utils.py:** Functions supporting gaze estimation and cursor control
- **keyboard.py:** Custom on-screen keyboard for demo videos

 This is followed by a brief explanation of the step-by-step script flow, the role of hyperparameters, and an explanation of the core code.



**STEP 0: FGI-Net model (fginet.py)**

FGI-Net is a lightweight, three-stage feature-fusion CNN designed for eye-gaze or coordinate regression. It enhances multi-scale features from EfficientNet and Swin using CBAM, fuses them in successive stages, and then predicts 2D coordinates via an MLP head.

**Input features (Input)**

- Size: `(B, 3, 224, 224)`
- Meaning: RGB image tensor

**Intermediate features by stage**

1. **Stage1** → `(B, 96, 56, 56)`
2. **Stage2** → `(B,168, 28, 28)`
3. **Stage3** → `(B,336, 14, 14)`

**Integrated features (Concat & Pooling)**

- Global average pooling of each stage output → `(B, C)` Convert to shape vector
- Concatenate the three to form a `(B, 96+168+336=600)` -dimensional vector – note that this deviates from the original FGI-Net and has been modified separately.

**Output features (Output)**

- Through the final MLP head `(B, 2)`
- Meaning: Predicted 2D coordinates (e.g. gaze position)

**Implementation FGI-Net model**

```python
class FGINet(nn.Module):
    """
    Main FGINet model: 3-stage feature fusion + MLP regression head.
    """

    def __init__(self):
        super().__init__()
        self.stage1 = Stage1_GIFModule(dropout_p=0.09)
        self.stage2 = Stage2_GIFModule(dropout_p=0.06)
        self.stage3 = Stage3_GIFModule(dropout_p=0.03)
        self.mlp_head = MLPHead(in_features=600, hidden_dim=32, out_dim=2, dropout_p=0.1)

    def forward(self, x):
        x1 = self.stage1(x)  # (B, 96, 56, 56)
        x2 = self.stage2(x)  # (B, 168, 28, 28)
        x3 = self.stage3(x)  # (B, 336, 14, 14)
        g1 = F.adaptive_avg_pool2d(x1, 1).flatten(1)
        g2 = F.adaptive_avg_pool2d(x2, 1).flatten(1)
        g3 = F.adaptive_avg_pool2d(x3, 1).flatten(1)
        feat = torch.cat([g1, g2, g3], dim=1)  # (B, 600)
        out = self.mlp_head(feat)  # (B, 2)
        return out
```



**STEP 1: csv data create (make_csv_custom.py)**

Follow the instructions to generate custom data (eye photo + X, Y coordinate labels) and create a csv file.

- **Settings block**

```python
# ========== [1] Configuration ==========
SAVE_DIR = "p01"  # Folder to save captured images
CSV_PATH = "p01.csv"  # Output label CSV file
N = 200  # Total number of target points
PATCH_PER_POINT = 10  # Number of images to save per point
ORDER_FILE = "targets_order.npy"  # File to save random order of targets
# ========================================
```

- **Collecting data**

When you run the file(make_csv_custom.py), points like the ones shown below will appear. While fixating on a point, pressing the space bar will capture 10 images at short intervals.

If you accidentally capture images without proper fixation, you can press the “z” key to cancel.

To ensure a well-distributed dataset, we recommend completing an entire target cycle before beginning training.

**Key bindings:**

- **Space**: Capture 10 images
- **Z**: Cancel the most recent capture
- **Esc**: Exit the application

![Image](https://github.com/user-attachments/assets/118ba8b7-b2f4-4087-8fb0-2df246507008)

```python
# Create directory to save images
os.makedirs(SAVE_DIR, exist_ok=True)

# Generate target points scaled to the screen dimensions
targets = make_targets(N, W, H)

# If a CSV with the same name exists, load existing data to continue collection
patches, labels = load_csv(CSV_PATH)

# If data collection was interrupted before covering all grid points, resume to ensure uniform distribution
    if os.path.exists(ORDER_FILE):
        targets_order, base_count = np.load(ORDER_FILE, allow_pickle=True)
        print(f"[Resume] Loaded random order ({len(targets_order)} points)")
        # Check for changed grid size/resolution
        if len(targets_order) != len(targets):
            print(f"[Warning] Current grid ({len(targets)}) and saved order ({len(targets_order)}) are different.")
            print("Complete your previous session before changing N/grid/settings.")
            exit(1)
        targets = targets_order
        base_count = int(base_count)
        idx = (len(patches) - base_count) // PATCH_PER_POINT
    else:
        random.shuffle(targets)
        base_count = len(patches)
        np.save(ORDER_FILE, np.array([targets, base_count], dtype=object))
        print(f"[New] Random order created ({len(targets)} pts), base_count={base_count}")
        idx = 0

# Crop only the eye regions from the webcam frame for training
patch = crop_eyes(frame, face_mesh, return_landmarks=False)

while True:  # Start the data collection loop
    if key == 27:  # Press ESC to terminate data collection
        pass
    elif key == ord(" "):  # Press Space to capture and save 10 images
        pass
    elif key == ord("z"):  # Press Z to delete the last 10 captures if miscaptured and prepare to retake
        pass

    # Save patches and labels to CSV after every 10 captures
    save_csv(patches, labels, CSV_PATH)

# After completing all grid points, remove the order file used for continuous collection
if os.path.exists(ORDER_FILE):
    os.remove(ORDER_FILE)
    print(f"All grid points complete! {ORDER_FILE} deleted.")
```

- The captured images are stored separately in the specified folder (e.g., `p01`).

  ![Image](https://github.com/user-attachments/assets/f99b1b95-a0bd-4fa3-a5f1-35a987f4b437)

- The saved CSV file has columns: `[image_path, dx, dy]`.

  ![Image](https://github.com/user-attachments/assets/3be8f0f7-a124-42e2-92aa-8871e03389a7)

**STEP 2: Model learning (train.py)**

Before running, update the CSV_PATH and CKPT_BEST settings at the top of train.py so they point to your CSV data file and to the location where the .pth weights will be stored, as needed.

Set the batch size to suit your own GPU (see the comments next to each setting for guidance).

Optionally, if you have a pretrained, generalized model (for example, one trained on MPIIGaze), you can fine-tune it on your own data.

- **Hyperparameters and settings blocks**

```python
# --- Config (with detailed comments) ---
CSV_PATH = "p01.csv"  # Path to training/inference CSV
PRETRAINED_PTH = "model_weights.pth"  # Initial (pretrained) weights
CKPT_BEST = "model_weights.pth"  # Path to save the best checkpoint
BATCH_SIZE = 16  # Mini-batch size
EPOCHS = 150  # Maximum number of epochs
SPLIT = (0.9, 0.05, 0.05)  # Data split ratio: (train, val, test)
SEED = 42  # Random seed for reproducibility
SPLIT_JSON = f"{SEED}_eye_patch_splits_train.json"  # Path to save split indices

LR_HEAD = 1e-3  # Learning rate for MLP head
LR_BACKBONE = 3e-4  # Learning rate for backbone
WARM_EPOCHS = 2  # Number of warmup epochs
PATIENCE = 30  # Early stop if no improvement for this many epochs
```

- **Model training, validation, and testing**

```python
# Split the CSV data into training, validation, and test sets, then create
# corresponding PyTorch DataLoader instances.
train_ld, val_ld, test_ld = get_loaders(CSV_PATH, SPLIT, BATCH_SIZE, SEED, SPLIT_JSON)

# Load pretrained weights if available
load_pretrained_weights(model, PRETRAINED_PTH, device)

# Resume training from a checkpoint if one exists
start_epoch, best = load_ckpt(model, opt, CKPT_BEST, device, LR_HEAD, LR_BACKBONE)

# Perform training and validation for one epoch, and record their losses
tr_loss = train_one_epoch(model, train_ld, opt, device, train=True)
vl_loss = train_one_epoch(model, val_ld, opt, device, train=False)

# Save the model checkpoint if validation loss improves to prevent overfitting
if vl_loss < best:
    best = vl_loss
    no_imp = 0
    save_ckpt(model, opt, ep + 1, best, CKPT_BEST)
    print("  ✓ best model saved")
else:
    # If validation loss does not improve for a set number of epochs, early stop
    no_imp += 1
    if no_imp >= PATIENCE:
        print(f"Early-stopped (no val improve {PATIENCE} ep)")
        break

# Evaluate the best model on the test set
model.load_state_dict(torch.load(CKPT_BEST, map_location=device)["model"])
model.eval()
test_loss = train_one_epoch(model, test_ld, None, device, train=False)
print(f"\n=== Test MAE (px): {test_loss:.2f}")
```



**STEP 3: Implementing gaze mouse with learned model(main.py)**

![Image](https://github.com/user-attachments/assets/82edb1ab-fdc6-4cb1-8b41-291661018954)

After completing training, run the `main.py` file to bring up a window like the one below.

Please refer to each parameter’s comment and tweak its value as needed:

- If the mouse doesn’t stay fixed well → increase the `FIX_RADIUS` value.
- If the mouse cursor is too jittery → increase the `SMOOTH_ALPHA` value.
- If, during calibration, the circle isn’t reliably detected as green → increase the `CALIB_STD_THRESHOLD` value.

After calibration completes, the mouse cursor will track your gaze and move accordingly. The click function works as follows:

- When you fix your gaze on a single point for a certain duration (default: 1.5 seconds), the cursor will lock in place.

- While locked, blinking both eyes will trigger a click.

- After clicking, the cursor unlocks immediately.

- If you don’t want to click, simply wait for the same duration (default: 1.5 seconds) without blinking.

The relevant parameters are:
```python
FIX_TIME = 1.5         # Time (sec) required to fixate before locking
UNLOCK_EYE_TIME = 1.5  # Time (sec) after fixation before unlocking by blink
```
![Image](https://github.com/user-attachments/assets/99c4dc28-b8c9-4c4a-ab44-1460ed514d27)

```cmd
[i] c: Calibrate | space: Load calibration | m: Move window | esc: Quit
```

**Key bindings:**

- **c**: Start calibration
- **Space**: Load existing calibration file
- **m**: Minimize the window and move it to the bottom-left
- **Esc**: Exit the application

**Initial preprocessing and manipulation**

- Calibration start screen![Image](https://github.com/user-attachments/assets/c578ddac-bdb8-46ea-840c-c204db02a487)

- Calibration screen

  | ![Image](https://github.com/user-attachments/assets/ca22ab2e-2588-4ff0-9718-7ce7f4d58c59) | ![Image](https://github.com/user-attachments/assets/d11c0165-0c87-48fe-a38d-314ba29a9a9e) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |

- **Settings block**

```python
# ----------------- Global Config -----------------
CALIB_FILE = "calib.npy"  # Calibration file path
CKPT = "model_weights.pth"  # Model checkpoint path
SMOOTH_ALPHA = 0.95  # Smoothing factor for gaze output (0~1, higher = smoother)
FIX_RADIUS = 60  # Radius (pixels) for fixation detection
FIX_TIME = 1.5  # Time (sec) required to fixate before locking
UNLOCK_EYE_TIME = 1.5  # Time (sec) after fixation before unlocking by blink
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
```

```python
# Load the trained model and the preprocessing transforms
model = load_model(CKPT, device)
tf = get_tf()

# Load the default calibration file if it exists; otherwise coefs will be None
coefs = load_calibration(calib_file)
calibrated = coefs is not None

while True:
    ret, frame = cap.read()                     # Read a frame from the webcam
    disp = cv2.resize(frame, CENTER_SIZE)       # Resize frame for display to the user

    if key == ord("c"):                         # Press 'c' to start calibration
        # Calibration function: since each user’s monitor size differs,
        # fit a linear mapping (first-order) generating two coefficients (x and y).
        coefs = calibrate(
            cap, model, tf, (W, H), cx, cy,
            device, face_mesh, std_threshold=CALIB_STD_THRESHOLD
        )

    elif key == ord(" "):                       # Press Space to load existing calibration
        coefs = load_calibration(calib_file)

    elif key == ord("m"):                       # Press 'm' to toggle the display size & position
        # Toggle small-display mode: keep the window always on top
        # and move it to the bottom-left corner to avoid occlusion
        show_small = not show_small

    elif key == 27:                             # Press ESC to exit
        break
```

- **Gaze inference and mouse movement**

```python
patch, lm_dict = crop_eyes(frame, face_mesh)  # Extract eye regions from the frame for inference
# Convert NumPy array to PIL image (BGR -> RGB)
pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

# Run model inference
with torch.no_grad():
    # Prepare input tensor: (C, H, W) -> add batch dim -> to GPU -> model -> back to CPU -> remove batch dim -> [dx, dy] -> to NumPy
    pred = model(tf(pil).unsqueeze(0).to(device)).cpu().squeeze().numpy()

# Apply smoothing filter for smoother cursor movement
smoothed = SMOOTH_ALPHA * prev + (1 - SMOOTH_ALPHA) * pred

if calibrated and coefs is not None:
    # Apply linear calibration if coefficients are available
    gaze_xy = apply_calib_linear(smoothed, coefs, cx, cy)
else:
    # Otherwise use raw model outputs
    gaze_xy = np.array([cx + smoothed[0], cy + smoothed[1]], dtype=np.float32)

# Clamp the gaze coordinates to stay within screen bounds
gx = int(np.clip(gaze_xy[0], 5, W - 5))
gy = int(np.clip(gaze_xy[1], 5, H - 5))
```

- **Eye blink Click**

```python
    # Save the current mouse position
    mouse_x, mouse_y = pyautogui.position()
    last_free_pos = (mouse_x, mouse_y)

    # Check if the mouse position remains within a specified radius (i.e., if the user is fixating)
    inside = is_inside_circle(fix_center, (mouse_x, mouse_y), FIX_RADIUS) if fix_center else False

    # Fixation logic
    if not fixed:
        if fix_center is None or not inside:  # If the point leaves the radius
            fix_center = (mouse_x, mouse_y)    # Update fixation center
            stay_timer = time.time()           # Reset dwell timer
        elif time.time() - stay_timer > FIX_TIME:  # If the point has been in place long enough
            fixed = True                      # Set fixation flag ON
            fixed_since = time.time()         # Record fixation start time (for auto-unlock)
            pyautogui.moveTo(*last_free_pos, _pause=False)  # Lock at the last free position

        if fixed:  # If in fixation state
            # If landmark data is available (i.e., both eyes detected in frame)
            if lm_dict and all(i in lm_dict for i in LEFT_EYE + RIGHT_EYE):
                # Compute eye aspect ratio for each eye
                l_ear = calculate_ear(LEFT_EYE, lm_dict)       # Left eye EAR
                r_ear = calculate_ear(RIGHT_EYE, lm_dict)      # Right eye EAR
                both_eyes_closed = l_ear < EAR_THRESHOLD and r_ear < EAR_THRESHOLD
            else:
                both_eyes_closed = False
            # Auto-unlock fixation after a set period
            if time.time() - fixed_since > UNLOCK_EYE_TIME:
                fixed = False
                fix_center = None
                fixed_since = None
                continue

            if both_eyes_closed:
                frame_counter += 1
            else:
                frame_counter = 0
            # If eyes remain closed for the required number of frames
            if frame_counter >= CONSEC_FRAMES:
                pyautogui.click()         # Perform a mouse click
                # Release fixation
                fixed = False
                fix_center = None
                frame_counter = 0

    # If fixated, keep cursor at the last free position
    if fixed and last_free_pos:
        pyautogui.moveTo(*last_free_pos, _pause=False)
    else:  # Otherwise, move cursor according to gaze
        pyautogui.moveTo(gx, gy, _pause=False)

```

- **User screen**

```python
    # Overlay a mask for user customization
    disp_to_show = draw_face_body_mask(frame, alpha=0.68)
    cv2.imshow(WIN_NAME, disp_to_show)
```

**(Added): Screen keyboard (keyboard.py)**

- If you run it in another terminal, the keyboard will appear. It is not an important file, so the explanation is omitted.

  

------

### 4. Results and Analysis

#### Result

- **Results screen**

  ![Image](https://github.com/user-attachments/assets/6c5a4939-4da1-4f69-a349-372950fcca9b)

- **MAE Loss**

  ![Image](https://github.com/user-attachments/assets/e05e429d-4325-4e69-9f14-25128eb40f6a)

  **Validation Set:** 66.77 (Pixel)

  **Test Set:** 67.71 (Pixel)

- **FPS**

  ![Image](https://github.com/user-attachments/assets/983fda21-2756-4cc1-ad90-ad45deb7f9c9)
  **Test FPS** : 22~24 fps

#### Analysis

|                     | Comparison model                                             | Improvement model                                            |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Execution screen    | ![Image](https://github.com/user-attachments/assets/96b3be63-18f3-49d2-8033-4650a75dfb8d) | ![Image](https://github.com/user-attachments/assets/c0a3d8f7-4492-40ab-b0df-bc47e23fab2b) |
| Accuracy (MAE loss) | 150~200 [Pixel]                                              | 60~70 [Pixel]                                                |
| FPS                 | 60 [fps]                                                     | 22~24 [fps]                                                  |

- **Comparison model** : As a comparison model, we used the AI-based webcam gaze and head-tracking software developed by Swiss company Eyeware Tech SA, which is currently the most widely used gaze-tracking program on the market.
- **Accuracy** : In the original program, the MAE loss was on the order of 150–200 pixels, resulting in a large discrepancy between the predicted gaze and the actual point on the screen—an error so great that precise mouse clicks were impossible. By contrast, our improved gaze-tracking model based on FGI-Net reduced the MAE loss to approximately 60 pixels, yielding an accuracy sufficient for reliable, precise clicking.
- **FPS** : The comparison model is capable of fast real-time tracking at around 60 FPS. By contrast, our accuracy-enhanced model runs at just over 20 FPS—still sufficient to avoid noticeable user discomfort, but a reduction in real-time tracking performance relative to the original.

#### Evaluation

- **Real-time** **performance**: decreased from 30 fps to 20 fps
- **Accuracy**: average MAE reduced from ~175 px to ~60 px → **≈66%** **error** **reduction**
- **Blink-based** **clicking**: enabled accurate hands-free clicks

#### Discussion

- Our evaluation shows a clear trade-off between tracking accuracy and real-time performance when comparing the Eyeware Tech SA solution to our FGI-Net–based model. The baseline system leverages an optimized AI pipeline to deliver fast, smooth tracking at around 60 FPS with very low latency. However, this speed comes at the cost of spatial precision: a mean absolute error (MAE) of 150–200 pixels creates a substantial gap between the user’s actual gaze and the predicted point, making precise cursor control and clicking practically impossible. 
- In contrast, our enhanced model—built on the FGI-Net architecture and focused on global information fusion—reduces MAE to approximately 60 pixels, boosting spatial accuracy by over 60 %. This improvement enables users to target and click on-screen elements with confidence. The price of this gain in precision is a drop in frame rate to the low-20 FPS range. Nevertheless, subjective testing indicates that, in typical desktop scenarios, users do not experience noticeable lag or discomfort.
- The current system relies on MediaPipe running on the CPU to extract eye and facial landmarks, achieving only about 30 FPS. By foregoing the full MediaPipe pipeline and instead using OpenCV to quickly crop just the eye region from each frame, we can dramatically reduce computational overhead and push real-time performance above 60 FPS. Since eye-region cropping is a simple image operation, it can readily leverage hardware acceleration or multithreading—making it a key strategy for further boosting tracking throughput.
- Additionally, our current implementation triggers a right-click only when both eyes blink simultaneously. This causes errors in the gaze-tracking model and makes mouse control unreliable if only one eye blinks. Furthermore, the single-eye blink classifier is not operating robustly, so additional development is required. We anticipate that accurately distinguishing left-eye and right-eye blink classes would enable far more precise mouse-click interactions.

------

### 5. conclusion

​	This study confirmed a clear trade-off between spatial accuracy and real-time performance when comparing the Eyeware Tech SA webcam gaze-tracking solution to our FGI-Net–based model: FGI-Net reduced MAE to about 60 pixels—a >60% error reduction—enabling precise clicking, but at the cost of frame rates dropping from 60 FPS to the low-20 FPS range. Nevertheless, users did not perceive lag in typical desktop scenarios, making FGI-Net ideal for applications demanding high precision, while the 60 FPS baseline remains preferable for smooth navigation in gaming or VR. Future work will replace MediaPipe with OpenCV–based eye-region cropping to restore performance above 60 FPS and refine left-/right-eye blink classification, thereby maximizing both accuracy and throughput in a unified approach.

### 6. References

[1] Zhang, C., & Wang, Y. (2024, November 27). *Lightweight Gaze Estimation Model Via Fusion Global Information* (Version 1) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2411.18064 [arxiv.org](https://arxiv.org/abs/2411.18064v1)

[2] Practical-CV. (n.d.). *EYE-BLINK-DETECTION-WITH-OPENCV-AND-DLIB* [Source code]. GitHub. Retrieved June 19, 2025, from https://github.com/Practical-CV/EYE-BLINK-DETECTION-WITH-OPENCV-AND-DLIB