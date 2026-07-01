# DreamDiffusion — BCS Summer Project 2025

> **Generating high-quality images directly from raw EEG brain signals using Temporal Masked Signal Modelling, CLIP alignment, and Stable Diffusion.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Diffusers-yellow?logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/CLIP-OpenAI-412991?logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/IIT%20Kanpur-BCS%20Summer%202025-orange" />
</p>

---

## Overview

This project implements and fine-tunes the **DreamDiffusion** pipeline as part of the **Brain and Cognitive Society (BCS) Summer Project 2025** at IIT Kanpur.

DreamDiffusion is a brain-to-image generation framework that synthesizes visual images directly from **EEG (electroencephalogram)** signals recorded while subjects viewed images from the ImageNet dataset. Rather than relying on text prompts, the model conditions a pre-trained Stable Diffusion model on learned EEG representations, bridging the gap between neural signals and visual semantics.

---

## Background & Motivation

EEG-based image generation is fundamentally difficult due to three core challenges:

| Challenge | Description |
|---|---|
| **Signal Noise** | EEG recordings contain significant artefacts from muscle movement, eye blinks, and electrical interference |
| **Low Information Density** | EEG captures only a coarse, averaged view of neural activity, unlike fMRI which offers spatial precision |
| **Inter-Subject Variability** | Neural responses to identical stimuli differ dramatically across individuals, making generalisation hard |

DreamDiffusion addresses all three through a two-stage learning strategy.

---

## Architecture

### Stage 1 — EEG Representation Learning (TMSM)

The EEG encoder is pre-trained using **Temporal Masked Signal Modelling (TMSM)**, a self-supervised masked autoencoding objective over time-series EEG segments. This allows the encoder to learn rich, noise-robust temporal representations **without requiring large labelled EEG–image datasets**.

```
Raw EEG Signal
      │
      ▼
  [Masking ~75%]
      │
      ▼
  Transformer Encoder  ──▶  Reconstructed EEG
      │
      ▼
  EEG Embedding
```

### Stage 2 — CLIP Alignment & Diffusion Conditioning

The learned EEG embeddings are aligned to the **CLIP joint text-image embedding space** via a lightweight projector network trained with a contrastive loss. This alignment allows EEG features to be used directly as conditioning signals for Stable Diffusion.

```
EEG Embedding  ──▶  [CLIP Projector]  ──▶  CLIP Space
                                                 │
                     CLIP Image Encoder  ────────┘  (Supervisory Signal)
                                                 │
                                                 ▼
                              Stable Diffusion (EEG-conditioned)
                                                 │
                                                 ▼
                                        Generated Image
```

---

## Repository Structure

```
DreamDiffusion_BCS/
├── DreamDiffusion_Colab_Clean.ipynb  # End-to-end pipeline notebook (Kaggle ready)
├── DreamDiffusion.ipynb          # Original development notebook
├── RelatedPaper.pdf              # Original DreamDiffusion paper (Bai et al., 2023)
├── Week 2 Resources.docx         # BCS project study resources
├── .gitignore                    # Excludes large model weights and datasets
└── README.md
```

> **Note:** Model weights (`*.pth`) and the ImageNet image dataset are excluded from the repository due to file size constraints. See the [Setup](#setup) section for how to obtain them.

---

## Pipeline Walkthrough

The `DreamDiffusion_Colab_Clean.ipynb` notebook covers the pipeline optimized for Kaggle/Colab execution. Since `latest.pth` (the fine-tuned UNet and projector weights) is already provided, the pipeline focuses heavily on **Inference**.

1. **Setup & Verification** — Setting up the environment on Kaggle, verifying GPU availability (T4x2 recommended), and checking the dataset directory.
2. **EEG Preprocessing** — Loading, filtering, and standardizing raw EEG signals; applying the `eeg_5_95_std` normalization checkpoint.
3. **EEG to CLIP Embedding** — Passing the preprocessed EEG signals through the pre-trained `EEGToCLIPProjector` (`eeg_to_clip_projector_best (1).pth`).
4. **Stable Diffusion Inference** — Replacing the text-conditioning pathway in Stable Diffusion with the EEG projector output. Loading the pre-trained weights (`latest.pth`) and generating images from held-out EEG test signals.

*(Note: The notebook also contains the training pipeline cells, which can be run if you wish to resume training or train on new data.)*

---

## Setup

### Prerequisites

```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install ftfy regex tqdm wandb
pip install git+https://github.com/openai/CLIP.git
pip install mne numpy matplotlib pandas tensorflow
```

### Required Files (not in repo)

The following large files must be downloaded and hosted as a Kaggle Dataset (or Google Drive for Colab):

| File | Purpose | Size |
|---|---|---|
| `eeg_5_95_std.pth` | Pre-trained EEG encoder weights | ~3.1 GB |
| `latest.pth` | Fine-tuned Stable Diffusion checkpoint | ~4.1 GB |
| `eeg_to_clip_projector_best (1).pth` | Trained CLIP projector weights | ~6 MB |
| `clip_image_embeddings.pt` | Pre-computed CLIP image embeddings | ~4 MB |
| `imageNet_images/` | ImageNet stimulus images used during EEG recording | ~224 MB |

### Running the Notebook (Recommended: Kaggle)

Due to memory constraints and GPU quotas on Google Colab, **Kaggle** is highly recommended for running this pipeline.

1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook.
2. Set the Accelerator to **GPU T4 x2**.
3. Create a dataset named `dreamdiffusion-bcs` (e.g., uploaded by `vaibhavkalyan`) containing all the `.pth` files and the `imageNet_images` folder.
4. Import `DreamDiffusion_Colab_Clean.ipynb` into the notebook.
5. Add the dataset to your notebook environment.
6. Run the cells in order to perform inference. (For inference only, you can run cells 1-9 and then skip directly to the last cell).

---

## Key Concepts

| Term | Definition |
|---|---|
| **EEG** | Electroencephalogram — a non-invasive recording of electrical brain activity via scalp electrodes |
| **TMSM** | Temporal Masked Signal Modelling — a self-supervised pretext task that masks and reconstructs EEG time segments |
| **CLIP** | Contrastive Language-Image Pre-Training — OpenAI's model that maps images and text into a shared embedding space |
| **Stable Diffusion** | A latent diffusion model for high-quality image generation, conditioned on embeddings |
| **RAG (in context)** | The EEG embedding acts as a retrieved, grounding signal analogous to retrieved context in LLM RAG systems |

---

## References

- Bai, W., Wang, X., et al. *DreamDiffusion: Generating High-Quality Images from Brain EEG Signals.* arXiv:2306.16934, 2023.  
  → [Paper PDF](https://arxiv.org/pdf/2306.16934) | [arXiv](https://arxiv.org/abs/2306.16934)

---

## Author

**Vaibhav Kalyan** — Third Year Undergraduate, Department of Statistics and Data Science, IIT Kanpur  
[GitHub](https://github.com/VaibhavKalyan) · [Email](mailto:vkalyan24@iitk.ac.in)

*Developed as part of BCS (Brain and Cognitive Society) Summer Project 2025, IIT Kanpur.*
