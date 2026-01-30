# Revise Deep Learning and Computer Vision

A hands-on Jupyter notebook that revises core **deep learning** and **computer vision** concepts using **PyTorch** and **YOLO**, from building neural networks from scratch to object detection and custom model training. Designed for Google Colab and local execution.

---

## Overview

This repository contains a single comprehensive notebook (`Deep_Learning_and_Computer_Vision.ipynb`) with five main demonstrations and a section of practical tips for real-world computer vision. Each section is self-contained with clear comments and can be run in order or independently (with minor setup).

---

## What the Code Covers

### Demonstration 1 — Building a Neural Network from Scratch

- **Goal:** Implement a feedforward neural network without using pre-built high-level APIs for classification.
- **Dataset:** MNIST (28×28 grayscale digits).
- **Model:** `SimpleNN` — 3 fully connected layers (784 → 128 → 64 → 10) with ReLU.
- **Concepts:** Data loading with `torchvision`, `DataLoader`, loss (`CrossEntropyLoss`), optimizer (Adam), training loop, and evaluation.
- **Outcome:** ~97% test accuracy on MNIST, establishing the basic PyTorch training pattern.

### Demonstration 2 — Convolutional Neural Network (CNN) for Image Classification

- **Goal:** Move from fully connected to convolutional architecture for better image handling.
- **Model:** `SimpleCNN` — two Conv2d blocks (with ReLU and MaxPool2d) followed by two linear layers.
- **Concepts:** `nn.Conv2d`, `nn.MaxPool2d`, feature map shapes, flattening before FC layers.
- **Outcome:** Same MNIST setup with a CNN; typically lower loss and similar or better accuracy than Demo 1.

### Demonstration 3 — Transfer Learning and Training Best Practices

- **Goal:** Introduce checkpointing, resuming training, and proper evaluation metrics.
- **Concepts:**
  - **Checkpoints:** Saving and loading `model_state_dict`, `optimizer_state_dict`, epoch, and loss with `torch.save()` / `torch.load()`.
  - **Resuming training:** Loading a checkpoint and continuing from a given epoch.
  - **Metrics:** Accuracy, Precision, Recall, and F1-Score using `sklearn.metrics`.
- **Outcome:** Same `SimpleNN` on MNIST, with periodic checkpoints and multi-metric evaluation (e.g. ~96% accuracy and F1).

### Demonstration 4 — Object Detection with YOLO

- **Goal:** Run pre-trained YOLO (Ultralytics YOLOv8) for object detection on images.
- **Libraries:** `ultralytics`, `opencv-python` (or `opencv-python-headless`), `matplotlib`, `PIL`.
- **Features in code:**
  - Load pre-trained model (e.g. `yolov8n.pt`).
  - Single image: upload or use a sample (e.g. from URL).
  - Batch processing for multiple images.
  - Detection from image URL.
  - Confidence threshold and class filtering.
  - Visualizing bounding boxes and saving annotated images.
- **Environment:** Written to run on **Google Colab** (uses `google.colab.files` for upload); can be adapted for local use by replacing upload logic.

### Demonstration 5 — Training YOLO on a Custom Dataset

- **Goal:** Train a YOLOv8 model on a custom (or built-in) dataset with error handling and verification.
- **Dataset:** Uses built-in **COCO8** for a quick test; the same pattern applies to your own YOLO-format dataset.
- **Steps in code:**
  - Install `ultralytics` and dependencies.
  - Load dataset config (e.g. `coco8.yaml`).
  - Load base model (`yolov8n.pt`), set epochs, batch size, image size, project name.
  - Run training, verify outputs and saved runs.
  - Load trained model, run inference, optionally export (e.g. ONNX) and download weights.
- **Outcome:** A trained YOLO model and saved weights, plus optional export for deployment.

### Practical Tips for Real-World Computer Vision

- **Tip 1 — Data augmentation:** Example pipeline with **Albumentations** (e.g. rotation, flips, blur, distortion, CLAHE, color).
- **Tip 2 — Learning rate scheduler:** Using `ReduceLROnPlateau` with validation loss on the same MNIST setup.
- **Tip 3 — Checkpoints:** Reinforces saving/loading checkpoints to resume training (aligned with Demo 3).
- **Tip 4 — Multiple metrics:** Reinforces tracking accuracy, precision, recall, and F1 during evaluation.

---

## Project Structure

```
Revise-Deep-Learning-and-Computer-Vision-main/
├── Deep_Learning_and_Computer_Vision.ipynb   # Main notebook (all demos + tips)
└── README.md                                  # This file
```

---

## Requirements and Dependencies

- **Python:** 3.8+ (tested with 3.12 on Colab).
- **Core (Demos 1–3, Tips):**
  - `torch`
  - `torchvision`
  - `scikit-learn`
  - `numpy`
- **Demo 4 & 5 (YOLO):**
  - `ultralytics`
  - `opencv-python` or `opencv-python-headless`
  - `matplotlib`
  - `Pillow`
- **Practical Tips:**
  - `albumentations` (for augmentation examples).

Install everything (e.g. local or Colab):

```bash
pip install torch torchvision scikit-learn numpy
pip install ultralytics opencv-python-headless Pillow matplotlib
pip install albumentations
```

For **Google Colab**, the notebook runs `!pip install ultralytics opencv-python-headless Pillow matplotlib` when needed; ensure the runtime has PyTorch (default Colab runtimes do).

---

## How to Run

### Option 1 — Google Colab (recommended)

1. Open the notebook in Colab using the **Open in Colab** badge/link at the top of the notebook (links to the GitHub repo).
2. Run cells in order. For Demo 4, use the Colab file upload or the provided URL option if you don’t upload.
3. For Demo 5, run the training cells; the first run will download COCO8 and the base YOLO model.

### Option 2 — Local

1. Clone the repo and open `Deep_Learning_and_Computer_Vision.ipynb` in Jupyter or VS Code.
2. Install the dependencies above (optionally in a virtual environment).
3. Run cells sequentially. For Demo 4, replace Colab-specific upload code with local paths or your own image-loading logic.
4. GPU is optional for Demos 1–3; helpful for Demo 5 (YOLO training).

---

## Key Implementation Details

| Topic              | Where in notebook | Main takeaway |
|--------------------|-------------------|----------------|
| NN from scratch    | Demo 1            | `nn.Module`, `forward()`, flatten 28×28 → 784, train/eval loop. |
| CNN                | Demo 2            | Conv → Pool → flatten → FC; same training pattern as Demo 1. |
| Checkpoints        | Demo 3, Tip 3     | Save state dicts + epoch/loss; load and resume optimizer + model. |
| Evaluation metrics | Demo 3, Tip 4     | Use `accuracy_score`, `precision_score`, `recall_score`, `f1_score` on predictions. |
| YOLO inference     | Demo 4            | `YOLO('yolov8n.pt')`, `model.predict()`, then use results for drawing/saving. |
| YOLO training      | Demo 5            | `model.train(data='coco8.yaml', epochs=..., ...)`, then validate and export. |
| Augmentation       | Tips              | Albumentations `Compose` with multiple transforms. |
| LR scheduler       | Tip 2             | `ReduceLROnPlateau` and `scheduler.step(val_loss)` after validation. |

---

## Datasets

- **MNIST:** Downloaded automatically by `torchvision.datasets.MNIST` (Demos 1–3 and Tips).
- **COCO8:** Downloaded automatically by Ultralytics when using `coco8.yaml` in Demo 5.
- **Custom YOLO:** For your own data, prepare a YOLO-format dataset and point the `data` argument in Demo 5 to your `.yaml` config.

---

## License and Attribution

- The notebook is part of the **Revise-Deep-Learning-and-Computer-Vision** repository.
- PyTorch and Ultralytics YOLO have their own licenses; see their official documentation.

---

## Quick Reference

- **Notebook:** `Deep_Learning_and_Computer_Vision.ipynb`
- **Colab:** Use the “Open in Colab” link in the notebook.
- **Install (minimal):** `pip install torch torchvision scikit-learn ultralytics opencv-python-headless matplotlib albumentations`

For a full revision path: run **Demo 1 → 2 → 3** for fundamentals, then **Demo 4 → 5** for object detection and custom training, and use the **Practical Tips** when building your own pipelines.
