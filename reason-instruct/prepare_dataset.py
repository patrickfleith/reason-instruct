import os
import sys
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import List
from dotenv import load_dotenv
load_dotenv()

# Dataset identifiers
IFEVAL_DATASET_ID = "argilla/ifeval-like-data"
IFEVAL_DEFAULT_SUBSET = "filtered"
TULU_DATASET_ID = "allenai/tulu-3-sft-personas-instruction-following"
ULTRAFEEDBACK_DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
NO_ROBOTS_DATASET_ID = "HuggingFaceH4/no_robots"

# Number of examples to select from each dataset
NUM_EXAMPLES = 2500
SEED = 42


def load_ifeval_dataset(
    subset: str = IFEVAL_DEFAULT_SUBSET,
    split: str = "train",
    cache_dir: str | None = None,
    shuffle: bool = True,
    seed: int = SEED,
    n: int | None = NUM_EXAMPLES
) -> Dataset:
    """
    Load the ifeval-like dataset from Argilla.
    
    Args:
        subset: The subset of the dataset to load
        split: The split of the dataset to load
        cache_dir: Directory to cache the dataset
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        n: Number of examples to select
        
    Returns:
        The loaded dataset with prompt and source_dataset_id columns
    """
    # Set default cache directory if None is provided
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(cache_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset(IFEVAL_DATASET_ID, subset, split=split, cache_dir=cache_dir)
    
    # Shuffle the dataset if requested
    if shuffle:
        dataset = dataset.shuffle(seed=seed).flatten_indices()
    
    # Select n instances if specified
    if n is not None:
        dataset = dataset.select(range(min(n, len(dataset))))
    
    # Add source_dataset_id column and keep only the required columns
    dataset = dataset.add_column("source_dataset_id", [IFEVAL_DATASET_ID] * len(dataset))
    dataset = dataset.select_columns(["prompt", "source_dataset_id"])
    
    return dataset


def load_tulu_dataset(
    split: str = "train",
    cache_dir: str | None = None,
    shuffle: bool = True,
    seed: int = SEED,
    n: int | None = NUM_EXAMPLES
) -> Dataset:
    """
    Load the tulu-3-sft-personas-instruction-following dataset from Allen AI.
    
    Args:
        split: The split of the dataset to load
        cache_dir: Directory to cache the dataset
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        n: Number of examples to select
        
    Returns:
        The loaded dataset with prompt and source_dataset_id columns
    """
    # Set default cache directory if None is provided
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(cache_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset(TULU_DATASET_ID, split=split, cache_dir=cache_dir)
    
    # Shuffle the dataset if requested
    if shuffle:
        dataset = dataset.shuffle(seed=seed).flatten_indices()
    
    # Select n instances if specified
    if n is not None:
        dataset = dataset.select(range(min(n, len(dataset))))
    
    # Add source_dataset_id column and keep only the required columns
    dataset = dataset.add_column("source_dataset_id", [TULU_DATASET_ID] * len(dataset))
    dataset = dataset.select_columns(["prompt", "source_dataset_id"])
    
    return dataset


def load_ultrafeedback_dataset(
    split: str = "train_prefs",  # Using train_prefs split as it contains prompts
    cache_dir: str | None = None,
    shuffle: bool = True,
    seed: int = SEED,
    n: int | None = NUM_EXAMPLES
) -> Dataset:
    """
    Load the UltraFeedback binarized dataset.
    
    Args:
        split: The split of the dataset to load (default: train_prefs)
        cache_dir: Directory to cache the dataset
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        n: Number of examples to select
        
    Returns:
        The loaded dataset with prompt and source_dataset_id columns
    """
    # Set default cache directory if None is provided
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(cache_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset(ULTRAFEEDBACK_DATASET_ID, split=split, cache_dir=cache_dir)
    
    # Shuffle the dataset if requested
    if shuffle:
        dataset = dataset.shuffle(seed=seed).flatten_indices()
    
    # Select n instances if specified
    if n is not None:
        dataset = dataset.select(range(min(n, len(dataset))))
    
    # Extract the prompt column and add source_dataset_id column
    dataset = dataset.add_column("source_dataset_id", [ULTRAFEEDBACK_DATASET_ID] * len(dataset))
    dataset = dataset.select_columns(["prompt", "source_dataset_id"])
    
    return dataset


def load_no_robots_dataset(
    split: str = "train",
    cache_dir: str | None = None,
    shuffle: bool = True,
    seed: int = SEED,
    n: int | None = NUM_EXAMPLES
) -> Dataset:
    """
    Load the No Robots dataset.
    
    Args:
        split: The split of the dataset to load
        cache_dir: Directory to cache the dataset
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        n: Number of examples to select
        
    Returns:
        The loaded dataset with prompt and source_dataset_id columns
    """
    # Set default cache directory if None is provided
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(cache_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset(NO_ROBOTS_DATASET_ID, split=split, cache_dir=cache_dir)
    
    # Shuffle the dataset if requested
    if shuffle:
        dataset = dataset.shuffle(seed=seed).flatten_indices()
    
    # Select n instances if specified
    if n is not None:
        dataset = dataset.select(range(min(n, len(dataset))))
    
    # Add source_dataset_id column and keep only the required columns
    dataset = dataset.add_column("source_dataset_id", [NO_ROBOTS_DATASET_ID] * len(dataset))
    dataset = dataset.select_columns(["prompt", "source_dataset_id"])
    
    return dataset


def combine_datasets(
    datasets: List[Dataset],
    shuffle: bool = True,
    seed: int = SEED
) -> Dataset:
    """
    Combine multiple datasets into one.
    
    Args:
        datasets: List of datasets to combine
        shuffle: Whether to shuffle the combined dataset
        seed: Random seed for shuffling
        
    Returns:
        The combined dataset
    """
    # Combine the datasets
    combined_dataset = concatenate_datasets(datasets)
    
    # Shuffle the combined dataset if requested
    if shuffle:
        combined_dataset = combined_dataset.shuffle(seed=seed).flatten_indices()
    
    return combined_dataset


if __name__ == "__main__":
    # Set output directory
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v2", "data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading {NUM_EXAMPLES} examples from each dataset with seed {SEED}...")
    
    # Load samples from each dataset
    ifeval_ds = load_ifeval_dataset(n=NUM_EXAMPLES, seed=SEED)
    tulu_ds = load_tulu_dataset(n=NUM_EXAMPLES, seed=SEED)
    ultrafeedback_ds = load_ultrafeedback_dataset(n=NUM_EXAMPLES, seed=SEED)
    no_robots_ds = load_no_robots_dataset(n=NUM_EXAMPLES, seed=SEED)
    
    # Combine the datasets
    combined_ds = combine_datasets([ifeval_ds, tulu_ds, ultrafeedback_ds, no_robots_ds], seed=SEED)
    
    print(f"Loaded {len(ifeval_ds)} examples from {IFEVAL_DATASET_ID}")
    print(f"Loaded {len(tulu_ds)} examples from {TULU_DATASET_ID}")
    print(f"Loaded {len(ultrafeedback_ds)} examples from {ULTRAFEEDBACK_DATASET_ID}")
    print(f"Loaded {len(no_robots_ds)} examples from {NO_ROBOTS_DATASET_ID}")
    print(f"Combined dataset contains {len(combined_ds)} examples")
    
    # Save the combined dataset to disk
    print(f"Saving combined dataset to {OUTPUT_DIR}...")
    combined_ds.save_to_disk(os.path.join(OUTPUT_DIR, "combined_dataset"))
    print(f"Dataset saved successfully!")
    
    # Print dataset statistics
    print("\nSources in combined dataset:")
    sources = [example["source_dataset_id"] for example in combined_ds]
    unique_sources = set(sources)
    for source in unique_sources:
        count = sources.count(source)
        print(f"{source}: {count} examples")
    
    # Push the dataset to the Hugging Face Hub
    # Check if HF_TOKEN is available
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("\nWarning: HF_TOKEN environment variable not found.")
        print("Please add your Hugging Face API token to the .env file:")
        print("HF_TOKEN=your_token_here")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)  # Exit with error code
        
    try:
        print("\nPushing dataset to the Hugging Face Hub...")
        # Define repository name in format: username/dataset-name
        repo_id = "patrickfleith/reasoning-instructions-mix"
        
        # Create dataset card with description
        dataset_card = f"""---
# Reasoning Instructions Mix

A combined dataset for fine-tuning reasoning models with high-quality instructions from multiple sources:

- {IFEVAL_DATASET_ID}: {sources.count(IFEVAL_DATASET_ID)} examples
- {TULU_DATASET_ID}: {sources.count(TULU_DATASET_ID)} examples
- {ULTRAFEEDBACK_DATASET_ID}: {sources.count(ULTRAFEEDBACK_DATASET_ID)} examples
- {NO_ROBOTS_DATASET_ID}: {sources.count(NO_ROBOTS_DATASET_ID)} examples

Total examples: {len(combined_ds)}

## Dataset Structure

Each example has two columns:
- `prompt`: The instruction or query
- `source_dataset_id`: The original dataset the example was sourced from
"""
        
        # Push to Hub
        combined_ds.push_to_hub(
            repo_id=repo_id,
            private=False,
            token=hf_token
        )
        
        # Update README separately using the huggingface_hub library
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=dataset_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            private=True
        )
        print(f"Dataset successfully pushed to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error pushing to Hugging Face Hub: {e}")
        print("Make sure you have set the HF_TOKEN environment variable with your Hugging Face API token.")