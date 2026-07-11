# 🎨 Pixel-Art Diffusion — DDPM from Scratch

A minimal, from-scratch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** in TensorFlow/Keras that learns to generate 32×32 pixel-art images.

> Based on the paper:  
> *Denoising Diffusion Probabilistic Models* — Ho, Jain & Abbeel, NeurIPS 2020  
> [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

---

## Overview

| | |
|---|---|
| **Task** | Unconditional image generation (pixel art, 32×32 RGB) |
| **Method** | DDPM — forward Gaussian noising + learned reverse denoising |
| **Model** | Lightweight U-Net with sinusoidal timestep conditioning |
| **Parameters** | ~308 K (~1.2 MB) |
| **Dataset** | [Pixel Art (Kaggle)](https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art) — ~2,794 images |
| **Framework** | TensorFlow 2 / Keras 3 |

---

## How Diffusion Works

```
Forward Process (training):                       Reverse Process (sampling):

  x₀ ──noise──▸ x₁ ──noise──▸ … ──noise──▸ x_T       x_T ──denoise──▸ x_{T-1} ──…──▸ x₀
  (clean)                              (pure noise)    (pure noise)                   (generated image)

  x_t = √ᾱ_t · x₀  +  √(1−ᾱ_t) · ε          The U-Net learns to predict ε given (x_t, t)
```

The model is trained to predict the noise `ε` that was added at each timestep. At inference time, we start from pure Gaussian noise and iteratively denoise using the trained model.

---

## Architecture

```
Inputs: [noisy_image (32,32,3)]  +  [timestep (scalar)]
                │                          │
                │                   TimestepEmbedding
                │                   (sinusoidal, dim=128)
                │                          │
         ┌──────┘                          │
         │                                 │
    ┌────▼─────────────────────────────────┤
    │  ENCODER                             │
    │  Conv2D(32)  → BN → +t_emb ──[skip1] │
    │  Conv2D(64,  s=2) → BN → +t_emb ──[skip2]
    │  Conv2D(128, s=2) → BN → +t_emb     │
    └────┬─────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────┐
    │  DECODER                             │
    │  ConvT(64,  s=2) → BN → +t_emb → Add(skip2)
    │  ConvT(32,  s=2) → BN → +t_emb → Add(skip1)
    │  ConvT(32)       → BN → +t_emb      │
    └────┬─────────────────────────────────┘
         │
    Conv2D(3) ── no activation ── predicted noise ε
```

---

## Project Structure

```
diffusion-main/
├── Diffusion.ipynb          # Main notebook — train & sample
├── README.md                # This file
├── RUNNING.md               # Step-by-step execution guide
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
│
│  (created after training)
├── diffusion_model.keras    # Saved full model
└── ckpt_best.weights.h5    # Best-epoch weight checkpoint
```

---

## Quick Start

```bash
# 1. Clone
git clone <repo-url> && cd diffusion-main

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open and run the notebook
jupyter notebook Diffusion.ipynb
```

For detailed step-by-step instructions, see [RUNNING.md](RUNNING.md).

---

## Notebook Cells at a Glance

| Cell | Purpose |
|---:|---|
| 1 | Title & description |
| 2 | Imports (`tensorflow`, `numpy`, `matplotlib`) |
| 3 | Noise schedule (linear β) + `add_noise` (DDPM forward process) |
| 4 | *(Optional)* Visualise forward diffusion on a sample image |
| 5 | U-Net model definition + sanity-check forward pass |
| 6 | Dataset download via `kagglehub` |
| 7 | `tf.data` pipeline — load, resize, normalise, noise, batch |
| 8 | Training — Adam + MSE loss, checkpointing, LR scheduler |
| 9 | Reverse diffusion sampling — generate & display pixel-art images |

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1 × 10⁻³ (halved on plateau, min 1 × 10⁻⁶) |
| Loss | Mean Squared Error (noise prediction) |
| Batch size | 32 |
| Diffusion steps (T) | 1,000 |
| β schedule | Linear, 0.0001 → 0.02 |
| Default epochs | 10 *(increase to 50–200 for better quality)* |

---

## Key Design Decisions

- **No activation on the output layer** — the model predicts noise `ε`, which is unbounded (can be negative). A `relu` or `sigmoid` here would break training.
- **Sinusoidal timestep embedding** — standard positional encoding from the original Transformer/DDPM papers. Allows the model to distinguish between early (low-noise) and late (high-noise) timesteps.
- **U-Net skip connections** — preserve spatial detail from the encoder so the decoder can reconstruct fine-grained structure.
- **Full T=999→1 reverse loop** — the sampling loop runs across all 1,000 diffusion steps for highest-quality generation.

---

## Results

After training, the model generates novel pixel-art images from pure Gaussian noise. Quality improves significantly with more training epochs:

| Epochs | Expected Quality |
|---|---|
| 10 | Blurry blobs with hints of colour structure |
| 50 | Recognisable pixel-art shapes and palettes |
| 100+ | Coherent pixel-art sprites |

---

## References

1. Ho, Jain & Abbeel — *Denoising Diffusion Probabilistic Models* (2020) — [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
2. Nichol & Dhariwal — *Improved Denoising Diffusion Probabilistic Models* (2021) — [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
3. Dataset — [Pixel Art on Kaggle](https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art)

---

## License

This project is provided for educational purposes. The pixel-art dataset is sourced from Kaggle under its respective license.
