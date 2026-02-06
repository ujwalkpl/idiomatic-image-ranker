# Idiomatic Image Ranking System

A multimodal image ranking system developed for **SemEval 2025 Task 1**, aimed at understanding idiomatic expressions in context and ranking relevant images accordingly.

## Problem Statement

Understanding idioms in a multimodal setting is a challenging task for NLP models. This project tackles **idiom-aware image ranking** - where for a given sentence containing an idiom, the model must rank 5 associated images by relevance.

Example:
> "It's raining cats and dogs" -> correct image: heavy rainfall, not actual cats or dogs.

## Project Structure

```
idiomatic-image-ranker/
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_loader.py       # Dataset classes and data loading
│   ├── model.py             # CLIP + LoRA model setup and loss functions
│   ├── training.py          # Training loop and logic
│   ├── evaluation.py        # Evaluation metrics (NDCG, Top-1 accuracy)
│   └── utils.py             # Utility functions (glossing, etc.)
├── scripts/                  # Executable scripts
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
├── data/                     # Dataset directory
│   ├── raw/                 # Raw image data
│   │   ├── train/
│   │   ├── dev/
│   │   ├── test/
│   │   └── xeval/
│   └── glosses/             # Gloss JSON files
├── outputs/                  # Training outputs
│   ├── checkpoints/         # Model checkpoints
│   ├── logs/                # Training logs (CSV)
│   └── plots/               # Plots and visualizations
├── notebooks/                # Jupyter notebooks
│   └── NLP_Project.ipynb   # Original research notebook
├── requirements.txt          # Python dependencies
├── setup.py                 # Package installation script
└── README.md                # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ujwalkpl/idiomatic-image-ranker.git
cd idiomatic-image-ranker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install the package in development mode:
```bash
pip install -e .
```

3. Set up environment variables (for gloss generation):
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

## Usage

### Training

Train a model with specific LoRA rank and loss function:

```bash
# Train with LoRA rank 1 and Euclidean distance loss
python scripts/train.py --lora-rank 1 --use-euclidean

# Train with LoRA rank 2 and cosine similarity loss
python scripts/train.py --lora-rank 2 --use-cosine

# Custom configuration
python scripts/train.py \
    --lora-rank 4 \
    --use-euclidean \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --max-epochs 25 \
    --patience 3
```

### Evaluation

Evaluate a trained model:

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/rank1_euclidean_best.pt \
    --lora-rank 1 \
    --use-euclidean \
    --split test

# Evaluate on dev set
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/rank1_euclidean_best.pt \
    --lora-rank 1 \
    --use-euclidean \
    --split dev
```

## Approach

### 1. **Gloss Generation**
- Used **DeepSeek LLM** to convert idiomatic sentences into their literal equivalents (glosses).

### 2. **Triplet Construction**
- Converted dataset into **(anchor, positive, negative)** triplets:
  - Anchor: gloss sentence  
  - Positive: correct image  
  - Negative: irrelevant image

### 3. **CLIP Fine-Tuning using LoRA**
- Fine-tuned CLIP's text and image encoders with **LoRA** (Low-Rank Adaptation).
- Compared **triplet loss using cosine similarity** and **Euclidean distance**.

## Experiments

- Varied **LoRA rank**: {1, 2, 4, 8}
- Tested with two loss functions:
  - Triplet Loss + Cosine Similarity
  - Triplet Loss + Euclidean Distance
- Metrics:
  - **Top-1 Accuracy**
  - **NDCG (Normalized Discounted Cumulative Gain)**

### Best Result
- **67% Top-1 Accuracy** using:
  - LoRA rank 1
  - Triplet Loss with Euclidean Distance

## Results Summary

| LoRA Rank | TLCS Test Acc | TLED Test Acc |
|-----------|---------------|---------------|
| 1         | 60%           | **67%**       |
| 2         | 40%           | 53%           |
| 4         | 40%           | 60%           |
| 8         | 53%           | 40%           |

## Training Details
- **Batch Size**: 16  
- **Learning Rate**: 1e-4  
- **Epochs**: max 25 (early stopping after 3)  
- **Optimizer**: Adam  

## Module Overview

### `src/data_loader.py`
Dataset classes for creating triplets from the idiom dataset.

### `src/model.py`
CLIP model setup with LoRA adapters and triplet loss functions.

### `src/training.py`
Training loop with early stopping and validation on dev set.

### `src/evaluation.py`
Evaluation metrics including NDCG and top-1 accuracy calculation.

### `src/utils.py`
Utility functions for gloss generation using DeepSeek LLM.

## Limitations
- Glosses may be ambiguous or inaccurate.
- Overfitting observed with higher LoRA ranks using cosine loss.
- Limited dataset size and idiom variety.

## Future Work
- Improve gloss generation quality.
- Extend to metaphors and multilingual idioms.
- Optimize model for deployment using quantization and distillation.

## Core Technologies
- **DeepSeek LLM** - Gloss generation  
- **CLIP + LoRA** - Efficient multimodal fine-tuning  
- **PyTorch** - Model implementation  
- **Triplet Loss** - Ranking-based training  

## Authors
Saivenu Kolli, Ujwal Karippali Chandran, Sanjay Baskaran, Arunava Ghosh  
University of Colorado Boulder

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{idiomatic-image-ranker,
  author = {Kolli, Saivenu and Chandran, Ujwal Karippali and Baskaran, Sanjay and Ghosh, Arunava},
  title = {Idiomatic Image Ranking System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ujwalkpl/idiomatic-image-ranker}
}
```
