# LAB: CNN Object Detection 2

## LAB: Eye tracking



### Introduction

With the recent advancements in AR/VR technologies, interest has grown in gaze-based control systems that function without dedicated hardware.

**Goal**

The goal of this project is to use a laptop’s built-in webcam to determine the user’s gaze and place the mouse pointer at the desired location based on that gaze. Existing solutions date back five or six years or are available only as applications without publicly available source code. Consequently, we plan to implement this system ourselves using the most up-to-date deep learning models possible.

### Problem Statement

Project objectives

- Enable gaze-based mouse control using only a laptop’s built-in webcam, without any specialized hardware.

- Improve upon models from five to six years ago by achieving both faster response time and higher accuracy.

Expected Outcome and Evaluation

**Cursor Accuracy**

- The mouse pointer should accurately appear at the exact screen location corresponding to the user’s gaze.
- A left-eye blink should trigger a left-click, and a right-eye blink should trigger a right-click.

**Quantitative Evaluation (RT-Gene Test Set)**

- Split the RT-Gene dataset into training, validation, and test subsets.
- Achieve an **L1 loss below 0.108** (i.e., below approximately 6.2°) on the test set.
- Maintain an inference speed of **30 frames per second (FPS) or higher** during real-time operation.

**User-Centric Evaluation**

- When an actual user performs gaze-based pointing and clicking, the cursor must follow the gaze with minimal lag and high precision, yielding a fluid, intuitive experience.

------

### Requirement

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

### Dataset

**RT-Gene** (Real-Time Gaze Estimation in Natural Environments) consists of around **92,000 face images** captured from 15 participants using standard webcams in an office/lab setting. Each participant performed typical desk tasks (reading, typing, looking around), resulting in varying lighting conditions, head poses (±40° yaw, ±40° pitch), and natural facial expressions. RT-Gene supplies RGB frames (cropped and resized to 224×224×3) paired with **2D gaze targets** (screen coordinates) and **3D head-pose angles** (Euler yaw, pitch). This “in-the-wild” indoor dataset allows evaluation of gaze models under more realistic conditions than strictly controlled lab data.

**Dataset link:**
 [RT-Gene on Zenodo](https://zenodo.org/records/2529036)

------

### Method



------

### Procedure

#### Installation

#### Tutorials

------

### Results and Analysis

### Reference
