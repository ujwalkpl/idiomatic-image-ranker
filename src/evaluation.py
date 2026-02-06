"""
Evaluation metrics for the idiomatic image ranker.
"""

import os
import ast
import numpy as np
import torch
from typing import Dict, Optional, List
import pandas as pd
from PIL import Image


def dcg(relevances: List[float]) -> float:
    """
    Calculate Discounted Cumulative Gain (DCG).
    
    Args:
        relevances: List of relevance scores in ranking order
        
    Returns:
        DCG score
    """
    relevances = np.asfarray(relevances)
    score = relevances[0]
    for i in range(1, len(relevances)):
        score += relevances[i] / np.log2(i + 2)
    return score


def ndcg_score(ideal_ranking: List[str], predicted_ranking: List[str]) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).
    
    Args:
        ideal_ranking: List of image names in ideal order
        predicted_ranking: List of image names in predicted order
        
    Returns:
        NDCG score between 0 and 1
    """
    # Map images to relevance scores (higher rank = higher relevance)
    image_to_relevance_score = {}
    for i in range(len(ideal_ranking)):
        image_to_relevance_score[ideal_ranking[i]] = len(ideal_ranking) - i 

    # Get relevance scores for predicted and ideal rankings
    predicted_relevance = []
    ideal_relevance = []
    
    for index in range(len(ideal_ranking)):
        predicted_relevance.append(image_to_relevance_score[predicted_ranking[index]])
        ideal_relevance.append(image_to_relevance_score[ideal_ranking[index]])
    
    # Calculate NDCG
    predicted_dcg = dcg(predicted_relevance)
    ideal_dcg = dcg(ideal_relevance)
    
    if ideal_dcg == 0:
        return 0.0
    
    return predicted_dcg / ideal_dcg


def rank_images(
    text: str,
    image_paths: Dict[str, str],
    model,
    processor,
    use_cosine: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[str]:
    """
    Rank images by their relevance to the text.
    
    Args:
        text: Text description
        image_paths: Dictionary mapping image names to file paths
        model: CLIP model
        processor: CLIP processor
        use_cosine: If True, use cosine similarity; else use Euclidean distance
        device: Device to run inference on
        
    Returns:
        List of image names sorted by relevance (most relevant first)
    """
    model.eval()
    
    with torch.no_grad():
        # Get text embedding
        text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        text_embedding = model.get_text_features(**text_inputs)
        
        # Calculate scores for each image
        scores = {}
        for image_name, image_path in image_paths.items():
            image = Image.open(image_path).convert("RGB")
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            image_embedding = model.get_image_features(**image_inputs)
            
            if use_cosine:
                # Higher cosine similarity = more relevant
                score = torch.nn.functional.cosine_similarity(
                    text_embedding, image_embedding
                ).item()
            else:
                # Lower Euclidean distance = more relevant
                score = -torch.nn.functional.pairwise_distance(
                    text_embedding, image_embedding, p=2
                ).item()
            
            scores[image_name] = score
    
    # Sort by score (descending)
    ranked_images = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [img_name for img_name, _ in ranked_images]


def calculate_ndcg_score(
    df: pd.DataFrame,
    data_dir: str,
    model,
    processor,
    use_cosine: bool = True,
    gloss_cache: Optional[Dict[str, str]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    """
    Calculate average NDCG score on a dataset.
    
    Args:
        df: DataFrame with columns 'sentence', 'expected_order', 'idiom'
        data_dir: Directory containing image data
        model: CLIP model
        processor: CLIP processor
        use_cosine: If True, use cosine similarity; else use Euclidean distance
        gloss_cache: Optional dictionary mapping sentences to glosses
        device: Device to run inference on
        
    Returns:
        Average NDCG score
    """
    model.eval()
    scores = []
    
    for index, row in df.iterrows():
        ideal_ranking = ast.literal_eval(row["expected_order"])
        
        # Use gloss if available
        if gloss_cache is None:
            text = row["sentence"]
        else:
            text = gloss_cache[row["sentence"]]
        
        # Build image paths
        image_to_image_paths = {}
        for image_name in ideal_ranking:
            image_to_image_paths[image_name] = os.path.join(
                data_dir, row["idiom"], image_name
            )
        
        # Rank images
        predicted_ranking = rank_images(
            text, image_to_image_paths, model, processor, 
            use_cosine=use_cosine, device=device
        )
        
        # Calculate NDCG
        score = ndcg_score(ideal_ranking, predicted_ranking)
        scores.append(score)
    
    return np.mean(scores)


def calculate_1pc_accuracy(
    df: pd.DataFrame,
    data_dir: str,
    model,
    processor,
    use_cosine: bool = True,
    gloss_cache: Optional[Dict[str, str]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    """
    Calculate top-1 accuracy on a dataset.
    
    Args:
        df: DataFrame with columns 'sentence', 'expected_order', 'idiom'
        data_dir: Directory containing image data
        model: CLIP model
        processor: CLIP processor
        use_cosine: If True, use cosine similarity; else use Euclidean distance
        gloss_cache: Optional dictionary mapping sentences to glosses
        device: Device to run inference on
        
    Returns:
        Top-1 accuracy (proportion of correct predictions)
    """
    model.eval()
    correct = 0
    
    for index, row in df.iterrows():
        ideal_ranking = ast.literal_eval(row["expected_order"])
        
        # Use gloss if available
        if gloss_cache is not None:
            text = gloss_cache[row["sentence"]]
        else:
            text = row["sentence"]
        
        # Build image paths
        image_to_image_paths = {}
        for image_name in ideal_ranking:
            image_to_image_paths[image_name] = os.path.join(
                data_dir, row["idiom"], image_name
            )
        
        # Rank images
        predicted_ranking = rank_images(
            text, image_to_image_paths, model, processor,
            use_cosine=use_cosine, device=device
        )
        
        # Check if top prediction is correct
        if predicted_ranking[0] == ideal_ranking[0]:
            correct += 1
    
    return correct / len(df)
