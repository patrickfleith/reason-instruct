import os
from datasets import load_dataset, Dataset
from typing import Optional

DATASET_ID = "argilla/ifeval-like-data"
DEFAULT_SUBSET = "filtered"


def load_ifeval_dataset(
    subset: str = DEFAULT_SUBSET,
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
    dataset = load_dataset(DATASET_ID, subset, split=split, cache_dir=cache_dir)
    
    # Shuffle the dataset if requested
    if shuffle:
        dataset = dataset.shuffle(seed=seed).flatten_indices()
    
    # Select n instances if specified
    if n is not None:
        dataset = dataset.select(range(min(n, len(dataset))))
        
    return dataset


# Example usage
if __name__ == "__main__":
    # Load a small sample of the dataset
    ds = load_ifeval_dataset(n=5)
    
    print(f"Loaded {len(ds)} examples from {DATASET_ID}")
    print("\nExample fields:")
    print(list(ds[0].keys()))
    
    print("\nFirst example input:")
    print(ds[0]["input"])