import os
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Optional, List, Dict, Any

# Dataset identifiers
IFEVAL_DATASET_ID = "argilla/ifeval-like-data"
IFEVAL_DEFAULT_SUBSET = "filtered"
TULU_DATASET_ID = "allenai/tulu-3-sft-personas-instruction-following"


def load_ifeval_dataset(
    subset: str = IFEVAL_DEFAULT_SUBSET,
    split: str = "train",
    cache_dir: Optional[str] = None,
    shuffle: bool = True,
    seed: int = 42,
    n: Optional[int] = None
) -> Dataset:
    """
    Load the ifeval-like dataset from Argilla.
    
    Args:
        subset: The subset of the dataset to load (default: "filtered")
        split: The split of the dataset to load (default: "train")
        cache_dir: Directory to cache the dataset (default: None, uses HF default)
        shuffle: Whether to shuffle the dataset (default: True)
        seed: Random seed for shuffling (default: 42)
        n: Number of examples to select (default: None, selects all)
        
    Returns:
        The loaded dataset
    """
    # Set default cache directory if None is provided
    if cache_dir is None:
        # Use the subdirectory "data" in the current directory
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        # Ensure the cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset(IFEVAL_DATASET_ID, subset, split=split, cache_dir=cache_dir)
    
    # Shuffle the dataset if requested
    if shuffle:
        dataset = dataset.shuffle(seed=seed).flatten_indices()
    
    # Select n instances if specified
    if n is not None:
        dataset = dataset.select(range(min(n, len(dataset))))
        
    # Add source column
    dataset = dataset.add_column("source", [IFEVAL_DATASET_ID] * len(dataset))
    
    return dataset


def load_tulu_dataset(
    split: str = "train",
    cache_dir: Optional[str] = None,
    shuffle: bool = True,
    seed: int = 42,
    n: Optional[int] = None
) -> Dataset:
    """
    Load the tulu-3-sft-personas-instruction-following dataset from Allen AI.
    
    Args:
        split: The split of the dataset to load (default: "train")
        cache_dir: Directory to cache the dataset (default: None, uses HF default)
        shuffle: Whether to shuffle the dataset (default: True)
        seed: Random seed for shuffling (default: 42)
        n: Number of examples to select (default: None, selects all)
        
    Returns:
        The loaded dataset
    """
    # Set default cache directory if None is provided
    if cache_dir is None:
        # Use the subdirectory "data" in the current directory
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        # Ensure the cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset(TULU_DATASET_ID, split=split, cache_dir=cache_dir)
    
    # Shuffle the dataset if requested
    if shuffle:
        dataset = dataset.shuffle(seed=seed).flatten_indices()
    
    # Select n instances if specified
    if n is not None:
        dataset = dataset.select(range(min(n, len(dataset))))
    
    # Add source column
    dataset = dataset.add_column("source", [TULU_DATASET_ID] * len(dataset))
        
    return dataset


def combine_datasets(
    datasets: List[Dataset],
    shuffle: bool = True,
    seed: int = 42
) -> Dataset:
    """
    Combine multiple datasets into one.
    
    Args:
        datasets: List of datasets to combine
        shuffle: Whether to shuffle the combined dataset (default: True)
        seed: Random seed for shuffling (default: 42)
        
    Returns:
        The combined dataset
    """
    # Combine the datasets
    combined_dataset = concatenate_datasets(datasets)
    
    # Shuffle the combined dataset if requested
    if shuffle:
        combined_dataset = combined_dataset.shuffle(seed=seed).flatten_indices()
    
    return combined_dataset


# Example usage
if __name__ == "__main__":
    # Set a fixed seed for reproducibility
    SEED = 42
    NUM_SAMPLES = 200
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "combined_dataset")
    
    # Load samples from both datasets with fixed seed for reproducibility
    print(f"Loading {NUM_SAMPLES} examples from each dataset with seed {SEED}...")
    ifeval_ds = load_ifeval_dataset(n=NUM_SAMPLES, seed=SEED)
    tulu_ds = load_tulu_dataset(n=NUM_SAMPLES, seed=SEED)
    
    # Combine the datasets
    combined_ds = combine_datasets([ifeval_ds, tulu_ds], seed=SEED)
    
    print(f"Loaded {len(ifeval_ds)} examples from {IFEVAL_DATASET_ID}")
    print(f"Loaded {len(tulu_ds)} examples from {TULU_DATASET_ID}")
    print(f"Combined dataset contains {len(combined_ds)} examples")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save the combined dataset to disk
    print(f"Saving combined dataset to {OUTPUT_DIR}...")
    combined_ds.save_to_disk(OUTPUT_DIR)
    print(f"Dataset saved successfully!")
    
    # Print some information about the dataset
    print("\nifeval Example fields:")
    print(list(ifeval_ds[0].keys()))
    
    print("\ntulu Example fields:")
    print(list(tulu_ds[0].keys()))
    
    print("\nFirst ifeval example prompt:")
    print(ifeval_ds[0]["prompt"][:200] + "..." if len(ifeval_ds[0]["prompt"]) > 200 else ifeval_ds[0]["prompt"])
    
    print("\nFirst tulu example prompt:")
    print(tulu_ds[0]["prompt"][:200] + "..." if len(tulu_ds[0]["prompt"]) > 200 else tulu_ds[0]["prompt"])
    
    print("\nSources in combined dataset:")
    sources = [example["source"] for example in combined_ds]
    unique_sources = set(sources)
    for source in unique_sources:
        count = sources.count(source)
        print(f"{source}: {count} examples")
