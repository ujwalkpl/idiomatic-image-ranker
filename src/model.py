"""
Model setup and configuration for CLIP with LoRA.
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig


def create_clip_model_with_lora(
    model_name: str = "openai/clip-vit-base-patch32",
    lora_rank: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Create a CLIP model with LoRA adapters.
    
    Args:
        model_name: HuggingFace model identifier
        lora_rank: Rank for LoRA adaptation (lower = fewer parameters)
        device: Device to load model on
        
    Returns:
        Tuple of (model, processor)
    """
    # Load base CLIP model and processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        target_modules=["q_proj", "v_proj"]
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    
    print(f"Model loaded with LoRA rank {lora_rank}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model, processor


def triplet_loss_cosine_similarity(
    anchor_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    negative_embedding: torch.Tensor,
    margin: float = 0.3
) -> torch.Tensor:
    """
    Triplet loss using cosine similarity.
    
    The loss encourages positive samples to have higher similarity to the anchor
    than negative samples, with a margin.
    
    Args:
        anchor_embedding: Text embeddings (anchor)
        positive_embedding: Positive image embeddings
        negative_embedding: Negative image embeddings
        margin: Minimum difference between positive and negative similarities
        
    Returns:
        Scalar loss value
    """
    pos_sim = torch.nn.functional.cosine_similarity(anchor_embedding, positive_embedding)
    neg_sim = torch.nn.functional.cosine_similarity(anchor_embedding, negative_embedding)
    loss = torch.relu(margin + neg_sim - pos_sim).mean()
    return loss


def triplet_loss_euclidean_distance(
    anchor_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    negative_embedding: torch.Tensor,
    margin: float = 0.3
) -> torch.Tensor:
    """
    Triplet loss using Euclidean distance.
    
    The loss encourages positive samples to be closer to the anchor than
    negative samples, with a margin.
    
    Args:
        anchor_embedding: Text embeddings (anchor)
        positive_embedding: Positive image embeddings
        negative_embedding: Negative image embeddings
        margin: Minimum difference between negative and positive distances
        
    Returns:
        Scalar loss value
    """
    pos_dist = torch.nn.functional.pairwise_distance(anchor_embedding, positive_embedding, p=2)
    neg_dist = torch.nn.functional.pairwise_distance(anchor_embedding, negative_embedding, p=2)
    loss = torch.relu(pos_dist - neg_dist + margin).mean()
    return loss
