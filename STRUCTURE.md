# Project Structure Documentation

This document explains the improved project structure and organization.

## Directory Layout

### `/src/` - Source Code Modules

Contains all reusable Python modules for the project:

- **`__init__.py`**: Package initialization
- **`data_loader.py`**: Dataset classes and data loading utilities
  - `TripletDataset`: Creates (anchor, positive, negative) triplets for training
  - `train_collate_fn`: Batch collation function for DataLoader
  
- **`model.py`**: Model architecture and loss functions
  - `create_clip_model_with_lora()`: Initialize CLIP with LoRA adapters
  - `triplet_loss_cosine_similarity()`: Triplet loss using cosine similarity
  - `triplet_loss_euclidean_distance()`: Triplet loss using Euclidean distance
  
- **`training.py`**: Training utilities
  - `train()`: Main training loop with early stopping and validation
  
- **`evaluation.py`**: Evaluation metrics
  - `calculate_ndcg_score()`: Calculate NDCG on dataset
  - `calculate_1pc_accuracy()`: Calculate top-1 accuracy
  - `rank_images()`: Rank images by relevance to text
  - `dcg()`: Discounted cumulative gain calculation
  - `ndcg_score()`: Normalized DCG calculation
  
- **`utils.py`**: Utility functions
  - `gloss_text()`: Convert idiomatic text to literal gloss using DeepSeek LLM
  - `preprocess_gloss()`: Batch process glosses for a dataset
  - `load_gloss_cache()`: Load pre-computed glosses from JSON
  - `save_gloss_cache()`: Save glosses to JSON

### `/scripts/` - Executable Scripts

Command-line scripts for training and evaluation:

- **`train.py`**: Training script with configurable hyperparameters
- **`evaluate.py`**: Evaluation script for trained models

### `/data/` - Datasets

Organized data storage:

- **`raw/`**: Raw image datasets
  - `train/`: Training set images (67 idioms)
  - `dev/`: Development set images (15 idioms)
  - `test/`: Test set images (16 idioms)
  - `xeval/`: Extended evaluation set images (all idioms)
  
- **`glosses/`**: Pre-computed gloss translations
  - `gloss_sentences_train.json`
  - `gloss_sentences_dev.json`
  - `gloss_sentences_test.json`

### `/outputs/` - Training Outputs

Generated during training and evaluation:

- **`checkpoints/`**: Saved model checkpoints (.pt files)
- **`logs/`**: Training logs and metrics (CSV files)
- **`plots/`**: Visualization plots (PNG files)

### `/notebooks/` - Jupyter Notebooks

- **`NLP_Project.ipynb`**: Original research notebook (kept for reference)

## Usage Examples

### Training a Model

```bash
# Train with LoRA rank 1 and Euclidean distance loss
python scripts/train.py --lora-rank 1 --use-euclidean

# Train with custom hyperparameters
python scripts/train.py \
    --lora-rank 2 \
    --use-cosine \
    --batch-size 32 \
    --learning-rate 5e-5 \
    --max-epochs 30 \
    --patience 5 \
    --model-name my_model
```

### Evaluating a Model

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/rank1_euclidean_best.pt \
    --lora-rank 1 \
    --use-euclidean \
    --split test
```

### Using as a Python Package

```python
from src.model import create_clip_model_with_lora
from src.data_loader import TripletDataset
from src.evaluation import calculate_ndcg_score

# Create model
model, processor = create_clip_model_with_lora(lora_rank=1)

# Load dataset
import pandas as pd
train_df = pd.read_csv("data/raw/train/subtask_a_train.tsv", sep='\t')
dataset = TripletDataset(train_df, data_dir="data/raw", split="train")

# Evaluate
ndcg = calculate_ndcg_score(
    train_df, 
    "data/raw/train", 
    model, 
    processor, 
    use_cosine=False
)
print(f"NDCG: {ndcg:.4f}")
```

## Benefits of New Structure

1. **Modularity**: Code is organized into logical modules that can be imported and reused
2. **Maintainability**: Easier to find and modify specific functionality
3. **Testability**: Each module can be tested independently
4. **Scalability**: Easy to extend with new features or models
5. **Clean Separation**: Clear separation between source code, data, outputs, and notebooks
6. **Professional**: Follows Python best practices and common project structures
7. **Documentation**: Clear README and structure documentation
8. **Package Installation**: Can be installed as a Python package with setup.py

## Migration from Old Structure

The old structure had all code in a single Jupyter notebook and files scattered in the root directory. The new structure:

- Extracted code into modular Python files in `src/`
- Moved all data to `data/raw/`
- Moved all outputs to `outputs/`
- Moved notebook to `notebooks/`
- Created executable scripts in `scripts/`
- Added proper .gitignore for Python projects
- Created setup.py for package installation
- Updated README with usage instructions

## Next Steps

To further improve the project:

1. Add unit tests in a `tests/` directory
2. Add a `docs/` directory with detailed documentation
3. Add CI/CD configuration (.github/workflows)
4. Add a `configs/` directory for experiment configurations
5. Add logging configuration
6. Add pre-commit hooks for code quality
7. Add type hints throughout the codebase
8. Add docstring tests
