#!/usr/bin/env python3
"""
Script to train the idiomatic image ranker model.
"""

import os
import sys
import argparse
import pandas as pd
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import create_clip_model_with_lora
from src.data_loader import TripletDataset
from src.training import train
from src.evaluation import calculate_ndcg_score
from src.utils import load_gloss_cache


def main():
    parser = argparse.ArgumentParser(description="Train idiomatic image ranker")
    parser.add_argument("--lora-rank", type=int, default=1, help="LoRA rank (default: 1)")
    parser.add_argument("--use-cosine", action="store_true", help="Use cosine similarity loss")
    parser.add_argument("--use-euclidean", action="store_true", help="Use Euclidean distance loss")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--max-epochs", type=int, default=25, help="Maximum epochs (default: 25)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (default: 3)")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument("--gloss-dir", type=str, default="data/glosses", help="Gloss cache directory")
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints", help="Output directory")
    parser.add_argument("--model-name", type=str, default=None, help="Model name prefix")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Determine loss type
    if args.use_euclidean:
        use_cosine = False
        loss_name = "euclidean"
    else:
        use_cosine = True
        loss_name = "cosine"
    
    # Set model name
    if args.model_name is None:
        args.model_name = f"rank{args.lora_rank}_{loss_name}"
    
    print(f"Training configuration:")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Loss type: {loss_name}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Model name: {args.model_name}")
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(os.path.join(args.data_dir, "train/subtask_a_train.tsv"), sep='\t')
    dev_df = pd.read_csv(os.path.join(args.data_dir, "dev/subtask_a_dev.tsv"), sep='\t')
    
    # Load gloss caches
    print("Loading gloss caches...")
    train_gloss = load_gloss_cache(os.path.join(args.gloss_dir, "gloss_sentences_train.json"))
    dev_gloss = load_gloss_cache(os.path.join(args.gloss_dir, "gloss_sentences_dev.json"))
    
    # Create model
    print("\nCreating model...")
    model, processor = create_clip_model_with_lora(lora_rank=args.lora_rank)
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = TripletDataset(
        train_df, 
        data_dir=args.data_dir,
        split="train",
        gloss_cache=train_gloss
    )
    print(f"Training samples: {len(train_dataset)}")
    
    # Train model
    print("\nStarting training...")
    logs = train(
        model=model,
        dataset=train_dataset,
        processor=processor,
        dev_df=dev_df,
        dev_gloss_cache=dev_gloss,
        evaluate_fn=calculate_ndcg_score,
        use_cosine=use_cosine,
        patience=args.patience,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
