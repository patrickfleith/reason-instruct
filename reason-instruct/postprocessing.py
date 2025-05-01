import os
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from datasets import Dataset, load_from_disk
from tqdm import tqdm
import json
from dotenv import load_dotenv
from bespokelabs import curator
import time

# Load environment variables for API keys if needed
load_dotenv()
# Set OpenAI API key if present in environment
if os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['CURATOR_DISABLE_CACHE'] = '1'


# Define validation models
class AnswerValidation(BaseModel):
    """Validation result for an answer."""
    is_valid: bool = Field(description="Whether the answer is valid with respect to the prompt")
    explanation: str = Field(description="Explanation of why the answer is valid or not")


ANSWER_VALIDATION_TEMPLATE = """
You are an expert at determining if an answer is a valid for a given user query.
Your task is to check if the provided answer is a valid response to the user query or if it contains errors message.

User query: {prompt}

Answer: {answer}

Is this a valid answer? (True/False)
"""

class AnswerValidatorLLM(curator.LLM):
    """
    A curator LLM class for extracting atomic instructions from user prompts.
    """
    response_format = AnswerValidation
    return_completions_object = False
    
    def prompt(self, input: Dict[str, str]) -> str:
        """
        Format the input prompt for the LLM using the answer validation template.
        """
        return ANSWER_VALIDATION_TEMPLATE.format(
            prompt=input["prompt"],
            answer=input["final_answer"]
        )

    def parse(self, input: Dict[str, str], response: AnswerValidation) -> list[Dict[str, object]]:
        """
        Parse the response from the LLM and return the answer validation result.
        Preserves the source_dataset_id column from the input dataset.
        """
        return [{
            "answer_validation": response.is_valid,
            "explanation": response.explanation
        }]


dataset = load_from_disk("results/reasoning_dataset_results")
# print("Dataset structure:", type(dataset))
# print("Dataset features:", dataset.features)
# print("Dataset columns:", dataset.column_names)

# Select columns correctly
prompts_and_answers = dataset.select_columns(["prompt", "final_answer"])
# print("Selected data sample:", prompts_and_answers[:2])

answer_validator = AnswerValidatorLLM(
    model_name="gpt-4o-mini"
)

# Run validation on the dataset
validated_data = answer_validator(prompts_and_answers)
print(f"Validation completed for {len(validated_data)} examples")

from datasets import concatenate_datasets

# Combine validation results with original dataset
new_ds = concatenate_datasets([dataset, validated_data], axis=1)

print("Example row before filtering:")
print(new_ds[0])

# Filter out invalid answers
filtered_ds = new_ds.filter(lambda example: example["answer_validation"] is True)
print(f"Filtered dataset from {len(new_ds)} to {len(filtered_ds)} rows")

# Remove validation columns
columns_to_remove = ["answer_validation", "explanation"]
filtered_ds = filtered_ds.remove_columns(columns_to_remove)

print("Example row after filtering and column removal:")
print(filtered_ds[0])

# Save dataset with validated answers
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "instruction-freak-reasoning")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save dataset locally
print(f"Saving dataset to {output_path}...")
filtered_ds.save_to_disk(output_path)
print(f"Dataset saved to {output_path}")

# Push to HuggingFace Hub
repo_id = "patrickfleith/instruction-freak-reasoning"
print(f"Pushing dataset to {repo_id}...")

try:
    filtered_ds.push_to_hub(
        repo_id=repo_id,
        private=True,
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Dataset successfully pushed to {repo_id}")
except Exception as e:
    print(f"Error pushing to hub: {e}")
    print(f"Dataset is still saved locally at: {output_path}")
