# Idiomatic Image Ranking System

A multimodal image ranking system developed for **SemEval 2025 Task 1**, aimed at understanding idiomatic expressions in context and ranking relevant images accordingly.

## Problem Statement

Understanding idioms in a multimodal setting is a challenging task for NLP models. This project tackles **idiom-aware image ranking** — where for a given sentence containing an idiom, the model must rank 5 associated images by relevance.

Example:
> "It’s raining cats and dogs" → correct image: heavy rainfall, not actual cats or dogs.

##  Approach

### 1. **Gloss Generation**
- Used **DeepSeek LLM** to convert idiomatic sentences into their literal equivalents (glosses).

### 2. **Triplet Construction**
- Converted dataset into **(anchor, positive, negative)** triplets:
  - Anchor: gloss sentence  
  - Positive: correct image  
  - Negative: irrelevant image

### 3. **CLIP Fine-Tuning using LoRA**
- Fine-tuned CLIP’s text and image encoders with **LoRA** (Low-Rank Adaptation).
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

##  Training Details
- **Batch Size**: 16  
- **Learning Rate**: 1e-4  
- **Epochs**: max 25 (early stopping after 3)  
- **Optimizer**: Adam  

## Limitations
- Glosses may be ambiguous or inaccurate.
- Overfitting observed with higher LoRA ranks using cosine loss.
- Limited dataset size and idiom variety.

## Future Work
- Improve gloss generation quality.
- Extend to metaphors and multilingual idioms.
- Optimize model for deployment using quantization and distillation.

## Core Technologies
- **DeepSeek LLM** – Gloss generation  
- **CLIP + LoRA** – Efficient multimodal fine-tuning  
- **PyTorch** – Model implementation  
- **Triplet Loss** – Ranking-based training  

## Authors
Saivenu Kolli, Ujwal Karippali Chandran, Sanjay Baskaran, Arunava Ghosh  
University of Colorado Boulder
