"""
Dataset classes for loading and preparing image ranking data.
"""

import os
import ast
from typing import Dict, Optional, Tuple
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    """
    Dataset that generates triplets (anchor, positive, negative) for training.
    
    Triplets consist of:
    - Anchor: A gloss sentence (literal meaning of idiomatic expression)
    - Positive: An image that correctly represents the idiom
    - Negative: An image that incorrectly represents the idiom
    """
    
    def __init__(self, df: pd.DataFrame, data_dir: str = "data/raw", 
                 split: str = "train", gloss_cache: Optional[Dict[str, str]] = None):
        """
        Args:
            df: DataFrame with columns 'sentence', 'expected_order', 'idiom'
            data_dir: Base directory containing the dataset
            split: Dataset split ('train', 'dev', or 'test')
            gloss_cache: Optional dictionary mapping sentences to glosses
        """
        self.anchor_positive_negative_triplets = []
        self.gloss_cache = gloss_cache
        self.data_dir = data_dir
        self.split = split

        for index, row in df.iterrows():
            expected_order = ast.literal_eval(row["expected_order"])
            
            # Use gloss if available, otherwise use original sentence
            if self.gloss_cache is not None:
                sentence = self.gloss_cache[row["sentence"]]
            else:
                sentence = row["sentence"]

            # Create triplets: positive is rank 1, negatives are ranks 2-5
            for i in range(1, 5):
                positive_image = expected_order[0]
                negative_image = expected_order[i]

                positive_path = os.path.join(
                    self.data_dir, 
                    self.split, 
                    row["idiom"], 
                    positive_image
                )
                negative_path = os.path.join(
                    self.data_dir, 
                    self.split, 
                    row["idiom"], 
                    negative_image
                )

                self.anchor_positive_negative_triplets.append(
                    (sentence, positive_path, negative_path)
                )

    def __len__(self) -> int:
        return len(self.anchor_positive_negative_triplets)

    def __getitem__(self, idx: int) -> Tuple[str, Image.Image, Image.Image]:
        """
        Returns:
            Tuple of (text, positive_image, negative_image)
        """
        sentence, positive_path, negative_path = self.anchor_positive_negative_triplets[idx]
        
        positive_image = Image.open(positive_path).convert("RGB")
        negative_image = Image.open(negative_path).convert("RGB")

        return sentence, positive_image, negative_image


def train_collate_fn(batch, processor):
    """
    Collate function for DataLoader to batch triplets.
    
    Args:
        batch: List of (text, positive_image, negative_image) tuples
        processor: CLIP processor for encoding text and images
        
    Returns:
        Tuple of (positive_inputs, negative_inputs)
    """
    texts = [item[0] for item in batch]
    pos_images = [item[1] for item in batch]
    neg_images = [item[2] for item in batch]

    inputs_pos = processor(
        text=texts, 
        images=pos_images, 
        return_tensors='pt', 
        padding=True, 
        truncation=True
    )
    inputs_neg = processor(
        text=texts, 
        images=neg_images, 
        return_tensors='pt', 
        padding=True, 
        truncation=True
    )
    
    return inputs_pos, inputs_neg
