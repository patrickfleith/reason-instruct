from typing import Dict, List
from pydantic import BaseModel, Field
from bespokelabs import curator
import os
from dotenv import load_dotenv
import prompts
from datasets import Dataset
from instruction_extractor import AtomicInstructions

# Load environment variables
load_dotenv()

# Set OpenAI API key if present in environment
if os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

os.environ['CURATOR_DISABLE_CACHE'] = '1'


class QueryAnalysis(BaseModel):
    explanation: str = Field(description="Explanation of what the user is asking")

class Answer(BaseModel):
    answer: str = Field(description="Answer to the user query")


class QueryAnalyzer(curator.LLM):
    response_format = QueryAnalysis
    
    def prompt(self, input: Dict) -> str:
        return prompts.QUERY_ANALYSIS_TEMPLATE.format(user_query=input['user_query'])
    
    def parse(self, input: Dict, response: QueryAnalysis) -> Dict:
        return [{'user_query': input['user_query'], 'explanation': response.explanation}]

class Responder(curator.LLM):
    response_format = Answer
    
    def prompt(self, input: Dict) -> str:
        return str(input['user_query'])
    
    def parse(self, input: Dict, response: Answer) -> Dict:
        return [{'user_query': input['user_query'], 'answer': response.answer}]

class InstructionExtractorLLM(curator.LLM):
    """
    A curator LLM class for extracting atomic instructions from user prompts.
    """
    response_format = AtomicInstructions
    return_completions_object = False
    
    def prompt(self, input: Dict) -> str:
        """
        Format the input prompt for the LLM using the instruction extraction template.
        """
        return prompts.INSTRUCTION_EXTRACTION_TEMPLATE.format(
            user_prompt=input['user_query']
        )
    
    def parse(self, input: Dict, response) -> Dict:
        """
        Parse the response from the LLM and return the extracted instructions.
        """
        return [{
            'user_query': input['user_query'],
            'atomic_instructions': response.instructions
        }]

from datafast.llms import OpenAIProvider, AnthropicProvider

query_analyzer = QueryAnalyzer(model_name="gpt-4o-mini")
responder = Responder(model_name="gpt-4o-mini")
instruction_extractor = InstructionExtractorLLM(model_name="gpt-4o")

user_queries = {
    "user_query": [
        # Regular instruction prompts
        "Create a web page that displays a list of products. Each product should have an image, name, price, and 'Add to Cart' button. When a user clicks on the button, the product should be added to a shopping cart.",
        "Write a function that takes a string as input and returns True if the string is a palindrome, False otherwise. A palindrome is a word that reads the same backward as forward.",
        # Multi-part question prompts
        "How does climate change affect biodiversity? Can you also explain the impact on ocean ecosystems?",
        # Complex multi-part questions
        "What is the relationship between quantum computing and cryptography, and how might quantum computers affect current encryption methods? Also, what timeline do experts predict for when quantum computers might break current encryption standards?",
        "How do different learning styles impact educational outcomes, and what teaching methods are most effective for visual, auditory, and kinesthetic learners? Additionally, how can teachers incorporate these different methods in a diverse classroom setting?",
        # Mixed genuine questions and disguised instructions
        "What is the capital of France? And could you also provide a list of the top 5 tourist attractions there?"
    ]
}

ds = Dataset.from_dict(user_queries)

print(ds, '\n\n')

query_dataset = query_analyzer(ds)
answer_dataset = responder(ds)

# Extract atomic instructions from the queries
instruction_dataset = instruction_extractor(ds)

# Save the resulting datasets to JSONL files
pandas_df = query_dataset.to_pandas()
pandas_df.to_json('queries.jsonl', orient='records', lines=True)
print(f"\nSaved {len(pandas_df)} queries to 'queries.jsonl'")

pandas_df = answer_dataset.to_pandas()
pandas_df.to_json('answers.jsonl', orient='records', lines=True)
print(f"\nSaved {len(pandas_df)} answers to 'answers.jsonl'")

pandas_df = instruction_dataset.to_pandas()
pandas_df.to_json('instructions.jsonl', orient='records', lines=True)
print(f"\nSaved {len(pandas_df)} instruction sets to 'instructions.jsonl'")



