#!/usr/bin/env python3
import os
import json
from tqdm import tqdm
from datasets import load_from_disk
from datetime import datetime
from typing import Dict, List, Any
import sys

from datafast.llms import OpenAIProvider, GeminiProvider
from dotenv import load_dotenv

# Import directly from the file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# We need to import the function directly since we can't import from a filename starting with a number
from typing import get_type_hints
import importlib.util

# Load the module directly using the full path
workflow_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_complete_workflow.py")
spec = importlib.util.spec_from_file_location("complete_workflow", workflow_path)
workflow_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow_module)

# Get the process_query function from the module
process_query = workflow_module.process_query

# Constants
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "combined_dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed_results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"processed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
MAX_ITERATIONS = 3
SAMPLE_SIZE = 100  # Set to a number if you want to process just a sample, or None for all

def load_combined_dataset():
    """Load the combined dataset from disk."""
    print(f"Loading dataset from {DATASET_PATH}...")
    
    try:
        dataset = load_from_disk(DATASET_PATH)
        print(f"Loaded {len(dataset)} examples successfully")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def count_refinements(reasoning_trace):
    """Count the number of refinements from the reasoning trace.
    
    Args:
        reasoning_trace: List of reasoning steps
        
    Returns:
        Number of refinements that took place
    """
    # Look for the pattern that indicates a refinement was made
    refinement_count = 0
    for step in reasoning_trace:
        if step.startswith("New Answer:"):
            refinement_count += 1
    
    return refinement_count


def process_dataset_items(dataset, sample_size=None):
    """Process each item in the dataset using the complete workflow.
    
    Args:
        dataset: The dataset to process
        sample_size: Optional number of examples to process (None = process all)
        
    Returns:
        List of processed results
    """
    # Sample dataset if requested
    if sample_size is not None and sample_size < len(dataset):
        dataset = dataset.select(range(min(sample_size, len(dataset))))
        
    results = []
    
    # Process each example with progress bar
    for item in tqdm(dataset, desc="Processing dataset"):
        try:
            # Get the prompt from the current dataset item
            user_prompt = item["prompt"]
            
            # Process the prompt using the complete workflow
            result = process_query(user_prompt, max_iterations=MAX_ITERATIONS)
            
            # Count the number of refinements from the reasoning trace
            refinement_count = count_refinements(result["reasoning_trace"])
            
            # Include source dataset information and refinement count in the result
            result["source_dataset"] = item["source"]
            result["num_refinements"] = refinement_count
            
            # Check if all instructions were satisfied (from the last reasoning trace entry)
            if result["reasoning_trace"] and "All instructions have been satisfied" in result["reasoning_trace"][-1]:
                result["all_instructions_satisfied"] = True
            else:
                result["all_instructions_satisfied"] = False
            
            # Add to results
            results.append(result)
            
            # Print occasional progress
            if len(results) % 5 == 0:
                print(f"Processed {len(results)} examples")
                
        except Exception as e:
            print(f"Error processing item: {e}")
            # Add a placeholder for failed items
            results.append({
                "user_query": item["prompt"],
                "error": str(e),
                "source_dataset": item["source"],
                "num_refinements": 0,
                "all_instructions_satisfied": False
            })
    
    return results

def save_results(results):
    """Save the processed results to a JSONL file.
    
    Args:
        results: List of processed results
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save results
    with open(OUTPUT_FILE, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"Saved {len(results)} processed results to {OUTPUT_FILE}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Load dataset
    dataset = load_combined_dataset()
    
    if dataset is not None:
        # Process dataset
        print(f"Starting to process {len(dataset) if SAMPLE_SIZE is None else SAMPLE_SIZE} examples...")
        results = process_dataset_items(dataset, sample_size=SAMPLE_SIZE)
        
        # Save results
        save_results(results)
        
        print("Processing complete!")
    else:
        print("Failed to load dataset. Please check the dataset path.")
