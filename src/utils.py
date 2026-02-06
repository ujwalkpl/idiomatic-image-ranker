"""
Utility functions for the idiomatic image ranker.
"""

import os
import re
import time
import json
import requests
from typing import Dict
import pandas as pd


def gloss_text(text: str, api_key: str, max_retries: int = 3, retry_delay: int = 2) -> str:
    """
    Convert idiomatic sentences into their literal equivalents (glosses) using DeepSeek LLM.
    
    Args:
        text: The sentence to gloss
        api_key: OpenRouter API key
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Glossed sentence or original text if glossing fails
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek/deepseek-r1-zero:free",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a language expert. Your task is to convert idiomatic or figurative language into literal explanations.\n"
                    "- If the sentence contains an idiom or figurative expression, rewrite it by replacing those parts with clear, literal meanings.\n"
                    "- If it doesn't, return the sentence unchanged.\n"
                    "DO NOT explain your reasoning or provide commentary. ONLY return the final rewritten sentence."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "reasoning": {
            "effort": "low",
            "exclude": True 
        }
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            if content:
                return content
            else:
                print(f"Attempt {attempt}: Empty content received. Retrying...")
                time.sleep(retry_delay)

        except Exception as e:
            print(f"Attempt {attempt}: Glossing failed due to error: {e}")
            time.sleep(retry_delay)

    print("Max retries reached. Returning original text.")
    return text


def preprocess_gloss(df: pd.DataFrame, api_key: str) -> Dict[str, str]:
    """
    Preprocess a dataframe by generating glosses for all unique sentences.
    
    Args:
        df: DataFrame containing sentences
        api_key: OpenRouter API key
        
    Returns:
        Dictionary mapping original sentences to their glosses
    """
    gloss_cache = {}
    for _, row in df.iterrows():
        if row["sentence"] not in gloss_cache:
            sentence = row["sentence"]
            gloss = gloss_text(sentence, api_key)
            # Extract from \boxed{...} if present
            match = re.search(r"\\boxed\{(.+?)\}", gloss)
            extracted = match.group(1).strip() if match else gloss.strip()
            gloss_cache[sentence] = extracted
            
    return gloss_cache


def load_gloss_cache(filepath: str) -> Dict[str, str]:
    """
    Load a gloss cache from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary mapping sentences to glosses
    """
    with open(filepath, "r") as file:
        return json.load(file)


def save_gloss_cache(gloss_cache: Dict[str, str], filepath: str) -> None:
    """
    Save a gloss cache to a JSON file.
    
    Args:
        gloss_cache: Dictionary mapping sentences to glosses
        filepath: Path to save the JSON file
    """
    with open(filepath, "w") as file:
        json.dump(gloss_cache, file, indent=4)
