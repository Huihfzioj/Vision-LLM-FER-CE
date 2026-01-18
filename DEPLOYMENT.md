# Deployment Guide

## Part 1: Push to GitHub
1.  **Create a Repository:** Go to GitHub.com and create a new repository (e.g., `Vision-LLM-FER`).
2.  **Push Code:** Run these commands in your local terminal:
    ```bash
    git add .
    git commit -m "Initial commit of Vision-LLM code"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/Vision-LLM-FER.git
    git push -u origin main
    ```

## Part 2: Running on Kaggle (Free T4/P100 GPU)

### A. Upload Data
Since `data/` is ignored (too large for Git), you must upload it separately.
1.  Go to **Kaggle Datasets** -> "New Dataset".
2.  Drag and drop your `data/` folder (zip it first if needed).
3.  Name it (e.g., `raf-ce-fer-dataset`).
4.  Create.

### B. Setup Notebook
1.  Create a **New Notebook** on Kaggle.
2.  **Settings (Right Sidebar):**
    *   Accelerator: **GPU T4 x2** (or P100).
    *   Internet: **On**.
3.  **Add Data:** Click "Add Input" -> "Your Datasets" -> Select `raf-ce-fer-dataset`.

### C. Run Code
In the first cell of your Kaggle Notebook:
```python
# 1. Clone Code
!git clone https://github.com/YOUR_USERNAME/Vision-LLM-FER.git
%cd Vision-LLM-FER

# 2. Install Deps
!pip install -r requirements.txt

# 3. Training
# NOTE: Update paths! Kaggle data is read-only at /kaggle/input
import sys
sys.path.append("src")
from models.vllm.blip2_model import Blip2Finetuner

# ... Copy your training script here ...
# IMPORTANT: Change "../data/train_instructions.json" to 
# "/kaggle/input/raf-ce-fer-dataset/data/train_instructions.json"
```

## Part 3: Running on Google Colab

1.  **Upload Code:**
    *   Easiest: `!git clone https://github.com/YOUR_USERNAME/Vision-LLM-FER.git`
    *   Or: Drag and drop the `src` folder to the left sidebar.
2.  **Upload Data:**
    *   Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
    *   Put your `data.zip` in Drive and unzip it in Colab: `!unzip /content/drive/MyDrive/data.zip`
3.  **Runtime:** Change Runtime Type -> T4 GPU.
4.  **Run:** (Same steps as Kaggle).
