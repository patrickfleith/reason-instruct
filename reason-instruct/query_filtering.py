from pydantic import BaseModel, Field
from bespokelabs import curator
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import prompts
from datasets import Dataset, load_dataset, load_from_disk
from semhash import SemHash
import sys

# Load environment variables
load_dotenv()

# Set OpenAI API key if present in environment
if os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def filter_by_length(dataset: Dataset, min_length: int = 20, max_length: int = 1000) -> Dataset:
    """
    Filter dataset to keep only instances with prompt length between min_length and max_length.
    
    Args:
        dataset: The dataset to filter
        min_length: Minimum number of characters required (default: 20)
        max_length: Maximum number of characters allowed (default: 1000)
        
    Returns:
        Filtered dataset
    """
    # Create a filter function to check prompt length is within range
    def is_proper_length(example):
        prompt_length = len(example["prompt"])
        return min_length <= prompt_length <= max_length
    
    # Apply the filter to the dataset
    filtered_dataset = dataset.filter(is_proper_length)
    print(f"Length filtering (between {min_length} and {max_length} chars): kept {len(filtered_dataset)}/{len(dataset)} examples ({(len(filtered_dataset)/len(dataset))*100:.2f}%)")
    return filtered_dataset


def remove_semantic_duplicates(dataset: Dataset, threshold: float = 0.5) -> Dataset:
    """
    Remove semantically similar examples using SemHash.
    
    Args:
        dataset: The dataset to deduplicate
        threshold: Similarity threshold for deduplication (default: 0.5)
        
    Returns:
        Deduplicated dataset
    """
    print(f"Starting semantic filtering on dataset with {len(dataset)} examples...")
    
    # Extract prompts for deduplication
    prompts = [str(p) for p in dataset["prompt"]]
    
    # Initialize the SemHash instance with the text data
    print("Initializing SemHash...")
    semhash = SemHash.from_records(records=prompts)
    
    # Perform semantic deduplication
    print("Performing semantic deduplication...")
    dedup_result = semhash.self_deduplicate(threshold=threshold)
    deduplicated_texts = dedup_result.selected
    print(f"Deduplication complete. Kept {len(deduplicated_texts)}/{len(prompts)} examples")
    
    # Create a set of deduplicated texts for quick lookup
    deduplicated_set = set(deduplicated_texts)
    
    # Create a filter function to keep only deduplicated examples
    def is_deduplicated(example: dict) -> bool:
        return str(example["prompt"]) in deduplicated_set
    
    # Apply the filter to the dataset
    final_dataset = dataset.filter(is_deduplicated)
    
    print(f"Overall: kept {len(final_dataset)}/{len(dataset)} examples ({(len(final_dataset)/len(dataset))*100:.2f}%)")
    return final_dataset


def apply_all_filters(dataset_path: str, output_path: str | None = None) -> Dataset:
    """
    Apply all filtering operations to the dataset.
    
    Args:
        dataset_path: Path to the dataset to filter
        output_path: Optional path to save the filtered dataset
        
    Returns:
        Filtered dataset
    """
    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    original_size = len(dataset)
    print(f"Original dataset size: {original_size} examples")
    
    # Apply character length filter (between 20 and 1000 characters)
    dataset = filter_by_length(dataset, min_length=20, max_length=1000)
    
    # Apply semantic deduplication
    dataset = remove_semantic_duplicates(dataset)
    
    # Print summary
    print(f"\nFiltering complete!")
    print(f"Original dataset: {original_size} examples")
    print(f"Filtered dataset: {len(dataset)} examples")
    print(f"Reduction: {(1 - len(dataset)/original_size)*100:.2f}%")
    
    # Save the filtered dataset if output path is provided
    if output_path:
        print(f"Saving filtered dataset to {output_path}...")
        dataset.save_to_disk(output_path)
        print("Dataset saved successfully!")
    
    return dataset


def push_filtered_dataset_to_hub(dataset_path: str, repo_id: str = "patrickfleith/reasoning-instructions-mix-filtered") -> None:
    """
    Push the filtered dataset to a dedicated Hugging Face Hub repository.
    
    Args:
        dataset_path: Path to the filtered dataset
        repo_id: Repository ID in format 'username/dataset-name'
    """
    # Check if HF_TOKEN is available
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("\nWarning: HF_TOKEN environment variable not found.")
        print("Please add your Hugging Face API token to the .env file:")
        print("HF_TOKEN=your_token_here")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)  # Exit with error code
    
    # Load the dataset
    print(f"Loading filtered dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Create dataset card with description
    dataset_card = f"""---
# Reasoning Instructions Mix - Filtered

A filtered version of the reasoning-instructions-mix dataset. This dataset has been:

1. Filtered by length (20-1000 characters)
2. Deduplicated semantically using SemHash (threshold=0.5)

Original size: 10,000 examples
Filtered size: {len(dataset)} examples
"""
    
    # Push to Hub as a dedicated repository
    print(f"\nPushing filtered dataset to the Hugging Face Hub as '{repo_id}'...")
    try:
        # Push the dataset to a separate repository
        dataset.push_to_hub(
            repo_id,
            token=hf_token,
            embed_external_files=False,
        )
        
        # Then upload README separately to add description
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=dataset_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
        )
        print(f"Dataset successfully pushed to {repo_id}!")
    except Exception as e:
        print(f"Error pushing dataset to Hub: {e}")



# Main execution
if __name__ == "__main__":
    # Path to the combined dataset from prepare_dataset.py
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "combined_dataset")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "filtered_dataset")
    
    # Apply all filters
    filtered_dataset = apply_all_filters(input_path, output_path)
    
    # Push the filtered dataset to Hugging Face Hub
    push_filtered_dataset_to_hub(output_path)


# Future implementation:
# Filtering with LLMs:
    # Keep english only
    # Number of instructions (use atomic instructions extractor)
    # Could this be a query a human would send to an AI assistant?