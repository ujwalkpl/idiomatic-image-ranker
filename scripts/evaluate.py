#!/usr/bin/env python3
"""
Script to evaluate the idiomatic image ranker model.
"""

import os
import sys
import argparse
import torch
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import create_clip_model_with_lora
from src.evaluation import calculate_ndcg_score, calculate_1pc_accuracy
from src.utils import load_gloss_cache


def main():
    parser = argparse.ArgumentParser(description="Evaluate idiomatic image ranker")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--lora-rank", type=int, default=1, help="LoRA rank (default: 1)")
    parser.add_argument("--use-cosine", action="store_true", help="Use cosine similarity")
    parser.add_argument("--use-euclidean", action="store_true", help="Use Euclidean distance")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "xeval"], 
                        help="Dataset split to evaluate on")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument("--gloss-dir", type=str, default="data/glosses", help="Gloss cache directory")
    
    args = parser.parse_args()
    
    # Determine loss type
    if args.use_euclidean:
        use_cosine = False
        loss_name = "euclidean"
    else:
        use_cosine = True
        loss_name = "cosine"
    
    print(f"Evaluation configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Loss type: {loss_name}")
    print(f"  Split: {args.split}")
    
    # Load data
    print("\nLoading data...")
    if args.split == "xeval":
        data_file = "subtask_a_xe.tsv"
    else:
        data_file = f"subtask_a_{args.split}.tsv"
    
    df = pd.read_csv(
        os.path.join(args.data_dir, args.split, data_file), 
        sep='\t'
    )
    
    # Load gloss cache
    print("Loading gloss cache...")
    gloss_file = f"gloss_sentences_{args.split}.json"
    gloss_cache = load_gloss_cache(os.path.join(args.gloss_dir, gloss_file))
    
    # Create model
    print("\nCreating model...")
    model, processor = create_clip_model_with_lora(lora_rank=args.lora_rank)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint))
    
    # Evaluate
    print("\nEvaluating...")
    data_path = os.path.join(args.data_dir, args.split)
    
    ndcg = calculate_ndcg_score(
        df, data_path, model, processor,
        use_cosine=use_cosine, gloss_cache=gloss_cache
    )
    
    accuracy = calculate_1pc_accuracy(
        df, data_path, model, processor,
        use_cosine=use_cosine, gloss_cache=gloss_cache
    )
    
    print("\nResults:")
    print(f"  NDCG: {ndcg:.4f}")
    print(f"  Top-1 Accuracy: {accuracy*100:.2f}%")
    
    # Save results
    results_file = os.path.join(
        "outputs/logs",
        f"eval_{args.split}_{os.path.basename(args.checkpoint).replace('.pt', '')}.txt"
    )
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"LoRA rank: {args.lora_rank}\n")
        f.write(f"Loss type: {loss_name}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"NDCG: {ndcg:.4f}\n")
        f.write(f"Top-1 Accuracy: {accuracy*100:.2f}%\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
