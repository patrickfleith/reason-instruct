from typing import Dict, List
from pydantic import BaseModel, Field
from bespokelabs import curator
import os
import json
from dotenv import load_dotenv

# Import template from prompts.py
from prompts import INSTRUCTION_EXTRACTION_TEMPLATE
from datamodels import AtomicInstructions
# Load environment variables
load_dotenv()

# Set OpenAI API key if present in environment
if os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['CURATOR_DISABLE_CACHE'] = '1'

# Define structured output model


class InstructionExtractor(curator.LLM):
    """
    A curator LLM class for extracting atomic instructions from user prompts.
    """
    response_format = AtomicInstructions
    # Don't return the completions object as it causes parsing issues
    return_completions_object = False

    def prompt(self, input: Dict) -> str:
        """
        Format the input prompt for the LLM.
        """
        return INSTRUCTION_EXTRACTION_TEMPLATE.format(
            user_prompt=input["user_prompt"]
        )

    def parse(self, input: Dict, response) -> Dict:
        """
        Parse the response from the LLM and return the extracted instructions.
        Simplified to avoid parsing issues.
        """
        # Directly use the instructions from the response
        # The response_format will handle the structured output
        return [{
            'user_prompt': input['user_prompt'],
            'atomic_instructions': response.instructions
        }]


# Template is now imported from prompts.py


def extract_instructions(user_prompt: str, model_name: str = "gpt-4o") -> Dict:
    """
    Extract atomic instructions from a user prompt.
    
    Args:
        user_prompt: The user prompt to analyze
        model_name: The model to use (default: "gpt-4o")
        
    Returns:
        A dictionary containing atomic instructions
    """
    # Initialize the extractor with simpler parameters
    extractor = InstructionExtractor(model_name=model_name)
    
    # Create input data
    input_data = [{"user_prompt": user_prompt}]
    
    # Process the input
    result = extractor(input_data)
    
    # Return the extracted instructions
    return {
        "user_prompt": user_prompt,
        "atomic_instructions": result[0]["atomic_instructions"]
    }


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Extract atomic instructions from a user prompt")
    parser.add_argument("prompt", type=str, help="The user prompt to analyze")
    parser.add_argument("--model", type=str, default="gpt-4o", help="The model to use")
    parser.add_argument("--output-json", action="store_true", help="Output full result as JSON")
    
    args = parser.parse_args()
    
    # Extract instructions
    result = extract_instructions(
        user_prompt=args.prompt,
        model_name=args.model
    )
    
    # Print results
    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        print("\nExtracted Atomic Instructions:")
        for i, instruction in enumerate(result.get("atomic_instructions", []), 1):
            print(f"{i}. {instruction}")
