"""
Training utilities for the idiomatic image ranker.
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional, Callable

from .data_loader import train_collate_fn
from .model import triplet_loss_cosine_similarity, triplet_loss_euclidean_distance


def train(
    model,
    dataset,
    processor,
    dev_df: pd.DataFrame,
    dev_gloss_cache: dict,
    evaluate_fn: Callable,
    use_cosine: bool = True,
    patience: int = 3,
    max_epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    output_dir: str = "outputs/checkpoints",
    model_name: str = "model",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the CLIP model with triplet loss.
    
    Args:
        model: CLIP model with LoRA
        dataset: Training dataset
        processor: CLIP processor
        dev_df: Validation dataframe
        dev_gloss_cache: Gloss cache for validation set
        evaluate_fn: Function to evaluate model (returns NDCG score)
        use_cosine: If True, use cosine similarity; else use Euclidean distance
        patience: Early stopping patience
        max_epochs: Maximum number of epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        output_dir: Directory to save checkpoints
        model_name: Name prefix for saved models
        device: Device to train on
        
    Returns:
        Training logs as a list of dictionaries
    """
    os.makedirs(output_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Wrap collate_fn with processor
    def collate_fn(batch):
        return train_collate_fn(batch, processor)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    # Select loss function
    if use_cosine:
        loss_fn = triplet_loss_cosine_similarity
        loss_type = "cosine"
    else:
        loss_fn = triplet_loss_euclidean_distance
        loss_type = "euclidean"

    best_dev_ndcg = 0
    patience_counter = 0
    best_path = os.path.join(output_dir, f"{model_name}_best.pt")
    logs = []

    print(f"Training with {loss_type} loss")
    print(f"Device: {device}")

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{max_epochs}")

        for inputs_pos, inputs_neg in pbar:
            # Move inputs to device
            inputs_pos = {k: v.to(device) for k, v in inputs_pos.items()}
            inputs_neg = {k: v.to(device) for k, v in inputs_neg.items()}

            # Forward pass for positive pairs
            outputs_pos = model(**inputs_pos)
            text_emb_pos = outputs_pos.text_embeds
            image_emb_pos = outputs_pos.image_embeds

            # Forward pass for negative pairs
            outputs_neg = model(**inputs_neg)
            image_emb_neg = outputs_neg.image_embeds

            # Compute loss
            loss = loss_fn(text_emb_pos, image_emb_pos, image_emb_neg)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(loader)
        
        # Evaluate on dev set
        dev_ndcg = evaluate_fn(
            dev_df, 
            "data/raw/dev", 
            model, 
            processor,
            use_cosine=use_cosine, 
            gloss_cache=dev_gloss_cache
        )

        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Dev NDCG = {dev_ndcg:.4f}")

        # Log results
        logs.append({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "dev_ndcg": dev_ndcg
        })

        # Early stopping
        if dev_ndcg > best_dev_ndcg:
            best_dev_ndcg = dev_ndcg
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved with NDCG: {best_dev_ndcg:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Load best model
    model.load_state_dict(torch.load(best_path))
    print(f"Training complete. Best Dev NDCG: {best_dev_ndcg:.4f}")

    # Save logs
    log_path = os.path.join("outputs/logs", f"{model_name}_{loss_type}_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    pd.DataFrame(logs).to_csv(log_path, index=False)
    print(f"Training logs saved to {log_path}")

    return logs
