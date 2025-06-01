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

**Quantitative Evaluation (MPIIGaze Test Set)**

- Split the MPIIGaze dataset into training, validation, and test subsets.
- Achieve a mean squared error (MSE) on the test set of **0.02 or lower** (normalized coordinate units).
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
- **Torchvision**: 0.20.1
- **Torchaudio**: 2.5.1
- **OpenCV (opencv-python)**: 4.11.0.86
- **NumPy**: 2.2.6
- **timm**: 1.0.15
- **tqdm**: 4.67.1
- **CUDA Runtime**: 12.1.0
  

### Dataset

**MPIIGaze** is a large-scale, real-world gaze dataset collected from 15 participants using their own laptop webcams. It contains over 200,000 grayscale eye‐patch images (36×60 pixels) paired with normalized screen‐coordinate gaze labels (x,y) and head‐pose angles (pitch, yaw). Images were captured while users looked at randomized points on their screens under varying lighting and head positions. MPIIGaze enables training and evaluation of appearance‐based gaze estimation models that generalize to uncontrolled environments.

**Dataset link:** [MPIIGaze (kaggle)](https://www.kaggle.com/datasets/dhruv413/mpiigaze/data) 

------

### Method



------

### Procedure

#### Installation

#### Tutorials

------

### Results and Analysis

### Reference
