import os
import sys
from datasets import Dataset, load_from_disk, concatenate_datasets
from dotenv import load_dotenv
from bespokelabs import curator

# Import custom modules
from datamodels import AtomicInstructions
from prompts import INSTRUCTION_EXTRACTION_TEMPLATE

# Load environment variables
load_dotenv()

# Set OpenAI API key if present in environment
if os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['CURATOR_DISABLE_CACHE'] = '1'


class InstructionExtractor(curator.LLM):
    """
    A curator LLM class for extracting atomic instructions from user prompts.
    """
    response_format = AtomicInstructions
    return_completions_object = False
    
    def prompt(self, input: dict[str, str]) -> str:
        """
        Format the input prompt for the LLM using the instruction extraction template.
        """
        return INSTRUCTION_EXTRACTION_TEMPLATE.format(
            user_query=input["prompt"]  # Use "prompt" field from the filtered dataset
        )

    def parse(self, input: dict[str, str], response: AtomicInstructions) -> list[dict[str, object]]:
        """
        Parse the response from the LLM and return the extracted instructions.
        Preserves the source_dataset_id column from the input dataset.
        """
        return [{
            "prompt": input["prompt"],
            "atomic_instructions": response.instructions,
            "source_dataset_id": input.get("source_dataset_id", None)  # Preserve source_dataset_id if it exists
        }]


def push_processed_dataset_to_hub(dataset_path: str, repo_id: str = "patrickfleith/reasoning-instructions-atomic") -> None:
    """
    Push the processed dataset to a dedicated Hugging Face Hub repository.
    
    Args:
        dataset_path: Path to the processed dataset
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
    print(f"Loading processed dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Get some statistics for the dataset card
    instruction_counts = [len(example["atomic_instructions"]) for example in dataset]
    avg_instructions = sum(instruction_counts) / len(instruction_counts) if instruction_counts else 0
    max_instructions = max(instruction_counts) if instruction_counts else 0
    min_instructions = min(instruction_counts) if instruction_counts else 0
    
    # Create dataset card with description
    dataset_card = f"""---
# Reasoning Instructions with Atomic Instructions

A processed version of the reasoning-instructions-mix dataset with extracted atomic instructions.

This dataset contains:
- Original user prompts
- Extracted atomic instructions for each prompt

Dataset statistics:
- Number of examples: {len(dataset)}
- Average instructions per example: {avg_instructions:.2f}
- Maximum instructions in an example: {max_instructions}
- Minimum instructions in an example: {min_instructions}
"""
    
    # Push to Hub as a dedicated repository
    print(f"\nPushing processed dataset to the Hugging Face Hub as '{repo_id}'...")
    try:
        # Push the dataset to a separate repository
        dataset.push_to_hub(
            repo_id,
            token=hf_token,
            embed_external_files=False,
            private=True  # Make the dataset private
        )
        
        # Then upload README separately to add description
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=dataset_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"Dataset successfully pushed to {repo_id}!")
    except Exception as e:
        print(f"Error pushing dataset to Hub: {e}")


def main():
    # Path to the filtered dataset from query_filtering.py
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "filtered_dataset")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed_dataset_with_instructions")
    
    # Load the filtered dataset
    print(f"Loading filtered dataset from {input_path}...")
    filtered_dataset = load_from_disk(input_path)
    print(f"Dataset loaded with {len(filtered_dataset)} examples")
    
    # Initialize the instruction extractor
    extractor = InstructionExtractor(model_name="gpt-4o-mini")
    
    # Process the entire dataset with the instruction extractor
    print("Processing the entire dataset with instruction extractor...")
    print(f"Total examples to process: {len(filtered_dataset)}")
    
    # Process the entire dataset at once
    processed_dataset = extractor(filtered_dataset)
    
    # Print summary statistics
    print(f"\nProcessing complete!")
    print(f"Original dataset: {len(filtered_dataset)} examples")
    print(f"Processed dataset: {len(processed_dataset)} examples")
    
    # Calculate statistics on number of instructions
    instruction_counts = [len(example["atomic_instructions"]) for example in processed_dataset]
    avg_instructions = sum(instruction_counts) / len(instruction_counts) if instruction_counts else 0
    max_instructions = max(instruction_counts) if instruction_counts else 0
    min_instructions = min(instruction_counts) if instruction_counts else 0
    
    print(f"Average instructions per example: {avg_instructions:.2f}")
    print(f"Maximum instructions in an example: {max_instructions}")
    print(f"Minimum instructions in an example: {min_instructions}")
    
    # Save the processed dataset
    print(f"Saving processed dataset to {output_path}...")
    processed_dataset.save_to_disk(output_path)
    print("Dataset saved successfully!")
    
    # Push the processed dataset to Hugging Face Hub
    push_processed_dataset_to_hub(output_path)
    
    return processed_dataset


if __name__ == "__main__":
    main()
