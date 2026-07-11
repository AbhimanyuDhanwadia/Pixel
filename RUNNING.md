# Running the Diffusion Notebook

Step-by-step guide to get the DDPM pixel-art generator running from zero.

---

## Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.9+ | 3.10–3.12 recommended |
| pip | 21+ | Comes with Python |
| Jupyter | — | `notebook` or `lab` |
| GPU *(optional)* | CUDA 11.8+ / cuDNN 8.6+ | Training is ~10× faster on GPU |

> **Kaggle / Google Colab users:** All dependencies are pre-installed.  
> Just upload `Diffusion.ipynb` and run all cells.

---

## 1 — Environment Setup

### Option A: Local (pip)

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda create -n diffusion python=3.11 -y
conda activate diffusion
pip install -r requirements.txt
```

### Option C: Google Colab / Kaggle

No setup needed — just open the notebook and run.  
The dataset download cell uses `kagglehub` which works natively on Kaggle.

> **Colab note:** If running on Colab, you may need to install `kagglehub` first:
> ```python
> !pip install kagglehub
> ```

---

## 2 — Launch the Notebook

```bash
jupyter notebook Diffusion.ipynb
# or
jupyter lab Diffusion.ipynb
```

---

## 3 — Run the Cells (in order)

Execute each cell **top to bottom**. Here's what each cell does and what to expect:

### Cell 1 — Title
Markdown header. Nothing to run.

### Cell 2 — Imports
```
import tensorflow, numpy, matplotlib, math, os
```
Should complete instantly. If you get an `ImportError`, run:
```bash
pip install tensorflow numpy matplotlib
```

### Cell 3 — Noise Schedule
Defines the linear β schedule (T=1,000 steps) and the `add_noise()` function.  
**Expected output:**
```
✅ Noise schedule ready  (T=1000, β_start=0.0001, β_end=0.0200)
```

### Cell 4 — Visualisation *(optional)*
Displays an image at 5 different noise levels (t=0, 250, 500, 750, 999).  
Requires a JPEG file at `./DSC_7047.jpg`. If the file doesn't exist, it prints a warning and skips — **this is fine.**

### Cell 5 — U-Net Model
Builds the model and runs a sanity-check forward pass.  
**Expected output:**
```
✅ Model built successfully!
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
...
 Total params: 307,971 (1.17 MB)

✅ Forward pass OK — output shape: (2, 32, 32, 3)
```

### Cell 6 — Dataset Download
Downloads the [Pixel Art dataset](https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art) (~2,794 images).  
- **On Kaggle:** Uses local cache, instant.  
- **Elsewhere:** Downloads from Kaggle Hub. You may need to authenticate first:
  ```bash
  pip install kagglehub
  # Then set your Kaggle credentials:
  export KAGGLE_USERNAME="your_username"
  export KAGGLE_KEY="your_api_key"
  ```
  Get your API key from https://www.kaggle.com/settings → API → "Create New Token".

**Expected output:**
```
Dataset path: /path/to/pixel-art
```

### Cell 7 — Data Pipeline
Builds the `tf.data` pipeline: load → resize to 32×32 → normalise [0,1] → add DDPM noise → batch.  
**Expected output:**
```
✅ Dataset ready — 2794 images, batch size 32
```

### Cell 8 — Training 🏋️
Trains the model. Default is **10 epochs**. Edit the `EPOCHS` variable at the top of the cell to change this.

**Recommended epochs by goal:**

| Goal | Epochs | Approx. Time (GPU) | Approx. Time (CPU) |
|---|---|---|---|
| Quick test | 1–5 | 2–5 min | 15–30 min |
| Decent results | 50 | 30–60 min | 5–8 hrs |
| Best quality | 100–200 | 1–3 hrs | 12–24 hrs |

The cell will:
1. Train the model with Adam + MSE loss
2. Save the best weights to `./ckpt_best.weights.h5`
3. Save the full model to `./diffusion_model.keras`
4. Plot the training loss curve

**Expected output:**
```
Epoch 1/10
88/88 ━━━━━━━━━━━━━━━━━━━━ 45s 386ms/step - loss: 0.0XXX
...
✅ Model saved to ./diffusion_model.keras
```

### Cell 9 — Sampling 🎨
Runs the full DDPM reverse diffusion (999 → 1) to generate **8 pixel-art images** from pure noise.

> ⏱ This takes 1–3 minutes on GPU (the loop runs 999 sequential model calls).

**Expected output:** A row of 8 generated pixel-art images displayed inline.

---

## 4 — Output Files

After a successful run, your project directory will contain:

```
diffusion-main/
├── Diffusion.ipynb              # ← your notebook
├── diffusion_model.keras        # Full saved model (reload with tf.keras.models.load_model)
├── ckpt_best.weights.h5         # Best-epoch weights only
└── ...
```

---

## 5 — Reloading a Trained Model

To skip training and jump straight to sampling:

```python
import tensorflow as tf

model = tf.keras.models.load_model("./diffusion_model.keras")
# Then run the sampling cell (Cell 9)
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow` |
| `ModuleNotFoundError: kagglehub` | `pip install kagglehub` |
| `OSError: No such file ... pixel-art` | Check Kaggle credentials (see Cell 6 notes above) |
| `OOM / ResourceExhausted` on GPU | Reduce `BATCH_SIZE` in Cell 7 (try 16 or 8) |
| Training loss doesn't decrease | Train for more epochs; ensure images are normalised to [0,1] |
| Generated images are all grey/uniform | Model needs more training (try 50+ epochs) |
| `DSC_7047.jpg not found` warning | This is fine — Cell 4 is optional visualisation |

---

## Running on Colab (Quickstart)

1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Upload Notebook → select `Diffusion.ipynb`
3. Runtime → Change runtime type → **T4 GPU**
4. Run all cells (Runtime → Run all)
5. The dataset downloads automatically via `kagglehub`
