# AquaMonitor Classification - Deep Learning Project

This repository contains the implementation for classifying benthic macroinvertebrates using deep learning models, specifically **ConvNeXt** and **ResNet18** architectures. The project includes data handling, model training, fine-tuning, evaluation, and visualization of key metrics like accuracy and F1-score.

---

## 📚 Project Highlights

- Two architectures implemented:
  - **ConvNeXt Tiny** (with progressive fine-tuning)
  - **ResNet18** (training available via Colab link)
- Class imbalance handled with custom class weights.
- Advanced data augmentation tailored for underwater imagery.
- Learning rate scheduling and progressive layer unfreezing applied.
- Metric visualization: loss, accuracy, F1-score over epochs.
- Best model saving based on validation F1 score.
- Supports both **local GPU** and **Google Colab** environments.

---

## 🚀 How to Run on Google Colab

You can quickly train the models using Google Colab by following these steps:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install correct versions of PyTorch and dependencies
!pip uninstall -y torch torchvision
!pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
!pip install timm datasets pandas huggingface-hub matplotlib tqdm

# Clone the repository (GPU_machine branch)
!git clone -b GPU_machine https://github.com/uzayyildiztaskan/AquaMonitor-Classification-DL.git
%cd AquaMonitor-Classification-DL

# Prepare dataset cache folder
import os
os.makedirs('dataset/.cache/huggingface', exist_ok=True)

# Start training
!python src/train.py
```

---

## 🖥️ How to Run Training Locally

To train models locally:

```bash
cd src
python train.py
```

Ensure you have the following Python packages installed:

```
torch
torchvision
timm
datasets
pandas
huggingface-hub
matplotlib
tqdm
```

---

## 🔥 **Important Notes**

- **Model Submission Format:**

  - The best performing model file has been renamed to `model.py` to comply with competition submission rules.
  - Model checkpoints are saved containing **both **``** and extra metadata** (e.g., optimizer state, scheduler state, metrics).
  - Model loading and evaluation logic for the competition can be found at the very end of `model.py` .

- **Checkpoint Format:**

  - Each `.pt` file checkpoint includes:
    - Model state dict
    - Optimizer state dict
    - Scheduler state dict
    - Epoch, phase, stage information
    - Metrics tracking (losses, accuracy, F1 scores)

---

## 📄 ResNet18 Training Example

You can see an example ResNet18 training setup on Colab [here](https://colab.research.google.com/drive/1EXJWWwwXHKfGZor8B3qmE9Fvh5-Lv3YX#scrollTo=y9Qty84EF7k-).

---

---

## 📊 Metrics & Visualization

During training, the following are plotted and saved:

- Training & Validation Loss
- Training & Validation Accuracy
- Training & Validation Weighted F1-Score

Plots are saved automatically inside the output directory after each phase.

---

