from datafast.llms import OpenAIProvider, GeminiProvider, AnthropicProvider
from datasets import Dataset
import prompts
from dotenv import load_dotenv
import os
import random
import time
from typing import Any
from datamodels import AtomicInstructions, InstructionAnalysis, InstructionVerificationResult, InstructionVerificationResults
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize OpenAI providers
query_analyzer_provider = AnthropicProvider(model_id="claude-3-5-haiku-latest")
answer_provider = AnthropicProvider(model_id="claude-3-5-haiku-latest")
refinement_provider = OpenAIProvider(model_id="gpt-4.1")
verification_provider = OpenAIProvider(model_id="gpt-4.1")


def analyze_query(user_query: str) -> str:
    """Analyze user query to understand what the user is asking.
    
    Args:
        user_query: The query from the user
        
    Returns:
        Analysis of the query
    """
    prompt = prompts.QUERY_ANALYSIS_TEMPLATE.format(
        user_query=user_query
    )
    
    analysis = query_analyzer_provider.generate(prompt=prompt)
    return f"{analysis}"


def generate_initial_answer(atomic_instructions: AtomicInstructions) -> str:
    """Generate initial answer based on extracted atomic instructions.
    
    Args:
        atomic_instructions: The atomic instructions extracted from the user query
        
    Returns:
        Initial answer based on a subset of instructions
    """
    # Deliberately select only a subset of instructions
    # Always include the first instruction, then each additional instruction with 50% probability
    selected_instructions = []
    
    if atomic_instructions.instructions:
        # Always include the first instruction
        selected_instructions.append(atomic_instructions.instructions[0])
        
        # For each remaining instruction, include it with 50% probability
        for instr in atomic_instructions.instructions[1:]:
            if random.random() < 0.5:  # 50% chance
                selected_instructions.append(instr)
    
    # Convert selected instructions to a formatted prompt
    instructions_text = "\n".join([f"{instr}" for instr in selected_instructions])
    
    answer = answer_provider.generate(prompt=instructions_text)
    return answer


def refine_answer(user_query: str, candidate_answer: str, critique: str) -> str:
    """Refine answer based on critique.
    
    Args:
        user_query: The original user query
        candidate_answer: The current answer
        critique: Critique of the current answer
        
    Returns:
        Refined answer
    """
    refinement_prompt = prompts.ANSWER_REFINEMENT_TEMPLATE.format(
        user_query=user_query,
        candidate_answer=candidate_answer,
        critique=critique
    )
    
    refined_answer = refinement_provider.generate(prompt=refinement_prompt)
    return refined_answer


def verify_instruction(user_query: str, candidate_answer: str, instruction: str) -> InstructionAnalysis:
    """Verify a single instruction against a user query and candidate answer"""
    prompt = prompts.INSTRUCTION_VERIFICATION_TEMPLATE.format(
        user_query=user_query,
        candidate_answer=candidate_answer,
        instruction=instruction
    )
    
    return verification_provider.generate(
        prompt=prompt,
        response_format=InstructionAnalysis
    )


def verify_instructions(user_query: str, candidate_answer: str, atomic_instructions: AtomicInstructions) -> InstructionVerificationResults:
    """Verify all instructions in AtomicInstructions against a user query and candidate answer"""
    results = []
    satisfied_count = 0
    
    for instruction in atomic_instructions.instructions:
        analysis = verify_instruction(user_query, candidate_answer, instruction)
        
        if analysis.satisfied:
            satisfied_count += 1
            
        results.append(InstructionVerificationResult(
            instruction=instruction,
            analysis=analysis
        ))
    
    return InstructionVerificationResults(
        results=results,
        satisfied_count=satisfied_count,
        total_count=len(atomic_instructions.instructions)
    )


def generate_critique(verification_results: InstructionVerificationResults) -> str:
    """Generate a formatted critique string from the verification results.
    
    Args:
        verification_results: The results of instruction verification
        
    Returns:
        A formatted string containing a summary and detailed critique of each instruction
    """
    critique_parts = []
    
    # Add summary header
    critique_parts.append("\nLet's verify that the answer satisfies all instructions:")
    
    # Add detailed analysis for each instruction
    for i, result in enumerate(verification_results.results):
        critique_parts.append(f"\nInstruction {i+1}: {result.instruction}")
        critique_parts.append(f"Explanation: {result.analysis.explanation}")
        critique_parts.append(f"Status: {'✓ Satisfied' if result.analysis.satisfied else '✗ Not satisfied'}")
        
        if not result.analysis.satisfied:
            critique_parts.append(f"Recommendation: {result.analysis.recommendation}")
    
    # # Add satisfaction ratio
    # ratio = verification_results.satisfaction_ratio
    # critique_parts.append(f"Satisfaction ratio: {ratio:.2f} ({verification_results.satisfied_count}/{verification_results.total_count})")

    # Final assessment
    if verification_results.satisfaction_ratio == 1.0:
        critique_parts.append("\nOverall assessment: All instructions have been satisfied. I can now answer the user query.")
    else:
        critique_parts.append(f"\nOverall assessment: {verification_results.total_count - verification_results.satisfied_count} "  
                           f"instruction(s) need attention to fully satisfy the requirements.")
    
    return "\n".join(critique_parts)


def generate_reasoning_trace(
    user_query: str,
    atomic_instructions: AtomicInstructions,
    max_iterations: int = 3
) -> tuple[list[str], str, int]:
    """Generate complete reasoning trace for a user query.
    
    The trace includes:
    1. Query analysis
    2. Initial answer
    3. Instruction verification and refinement
    
    Args:
        user_query: The user's query
        max_iterations: Maximum number of refinement attempts
        
    Returns:
        Tuple containing (reasoning_trace, final_answer)
    """
    reasoning_trace = []
    
    # Step 1: Analyze the query
    query_analysis = analyze_query(user_query)
    reasoning_trace.append(query_analysis)
    
    # Step 2: Generate initial answer using the extracted instructions
    initial_answer = generate_initial_answer(atomic_instructions)
    
    reasoning_trace.append(f"I'll draft an initial answer based on all extracted instructions:\n{initial_answer}")
    
    # Step 3: Verify and refine
    current_answer = initial_answer
    iteration = 0
    all_satisfied = False

    while iteration < max_iterations:
        # Increment iteration counter
        iteration += 1
        
        # Verify current answer
        verification_results = verify_instructions(
            user_query=user_query,
            candidate_answer=current_answer,
            atomic_instructions=atomic_instructions
        )
        
        # Generate critique and add to trace
        critique = generate_critique(verification_results)
        reasoning_trace.append(critique)

        # Check if all instructions are satisfied
        if verification_results.satisfaction_ratio == 1.0:
            all_satisfied = True
            break
            
        # If we've reached max iterations, stop here (with the critique as the last item)
        if iteration >= max_iterations:
            break
            
        # Otherwise add refinement step to trace
        reasoning_trace.append("I need to refine the answer based on this assessment.")
            
        # Generate refined answer
        refined_answer = refine_answer(
            user_query=user_query,
            candidate_answer=current_answer,
            critique=critique
        )
        
        # Add the refined answer to trace and update for next iteration
        reasoning_trace.append(f"New Answer:\n{refined_answer}")
        current_answer = refined_answer

    # Add final assessment message to trace if not all instructions were satisfied
    if not all_satisfied:
        reasoning_trace.append(f"After {iteration} refinement attempts, {verification_results.total_count - verification_results.satisfied_count} "  
                     f"instruction(s) still need possible attention. I'll return the best available answer so far.")
    
    return reasoning_trace, current_answer, iteration


def format_reasoning_trace(reasoning_trace: list) -> str:
    """Format the reasoning trace list into a single well-formatted string.
    
    Args:
        reasoning_trace: List of reasoning trace items
        
    Returns:
        A single string containing the entire reasoning trace
    """
    # Using just a single line break between trace elements
    separator = "\n\n"
    
    # Join all items in the trace with the separator
    return separator.join(reasoning_trace)


def process_query(user_query: str, atomic_instructions: AtomicInstructions, max_iterations: int = 3) -> dict[str, Any]:
    """Process a single user query and return results.
    
    Args:
        user_query: The user's query
        atomic_instructions: The atomic instructions extracted from the user query
        max_iterations: Maximum number of refinement attempts
        
    Returns:
        Dictionary containing:
        - user_query: Original query
        - reasoning_trace: List of reasoning steps
        - reasoning_text: Formatted reasoning trace as text
        - final_answer: The final answer
        - refinement_count: Number of refinement iterations performed
    """
    reasoning_trace, final_answer, refinement_count = generate_reasoning_trace(
        user_query=user_query,
        atomic_instructions=atomic_instructions,
        max_iterations=max_iterations
    )
    
    return {
        "user_query": user_query,
        "atomic_instructions": atomic_instructions,
        "reasoning": format_reasoning_trace(reasoning_trace),
        "final_answer": final_answer,
        "num_iterations": refinement_count
    }


def process_dataset(dataset: Dataset, max_iterations: int = 3) -> list[dict[str, Any]]:
    """Process a list of queries and return results for each.
    
    Args:
        dataset: Dataset containing user queries to process
        max_iterations: Maximum number of refinement attempts per query
        
    Returns:
        List of result dictionaries, one per query
    """
    results = []
    error_count = 0
    
    for i, row in enumerate(tqdm(dataset)):
        time.sleep(1) # to avoid hitting rate limits
        try:
            # Use 'prompt' as the user query (not 'query')
            result = process_query(row["prompt"], AtomicInstructions(instructions=row["atomic_instructions"]), max_iterations)
            results.append(result)
        except Exception as e:
            error_count += 1
            print(f"\n❌ Error processing item {i}/{len(dataset)}: {e}")
            print(f"Problem item prompt: {row.get('prompt', '(prompt not available)')}\n")
            print(f"Waiting 60 seconds before continuing...")
            
            # Add a placeholder result with error information
            error_result = {
                "user_query": row.get("prompt", "Error: prompt not available"),
                "atomic_instructions": AtomicInstructions(instructions=["Error: processing failed"]),
                "reasoning": f"Error occurred during processing: {str(e)}",
                "final_answer": "Error: processing failed",
                "num_iterations": 0,
                "error": str(e)
            }
            results.append(error_result)
            
            # Sleep for 1 minute before continuing
            time.sleep(60)
    
    if error_count > 0:
        print(f"\nProcessing completed with {error_count} errors out of {len(dataset)} items")
    
    return results


# Example usage
if __name__ == "__main__":
    import os
    from datasets import load_from_disk, Dataset
    
    # Path to the processed dataset with instructions
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed_dataset_with_instructions")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "reasoning_dataset_results")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the dataset
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Set to None for full dataset processing
    max_examples = 200  # Process only 100 examples for submission
    if max_examples:
        dataset_subset = dataset.select(range(min(max_examples, len(dataset))))
        print(f"Processing subset of {len(dataset_subset)} examples...")
    else:
        dataset_subset = dataset
        print(f"Processing all {len(dataset_subset)} examples...")
    
    # Process the dataset
    results = process_dataset(dataset_subset)
    
    # Build a new dataset combining original data with results
    processed_data = {
        "prompt": [],                  # Original query/prompt 
        "source_dataset_id": [],      # Source dataset ID
        "atomic_instructions": [],     # List of atomic instructions
        "reasoning": [],              # Reasoning trace
        "final_answer": [],           # Final generated answer
        "num_iterations": []        # Number of refinement iterations performed
    }
    
    # Collect data from results and original dataset
    for i, result in enumerate(results):
        # Get source_dataset_id from original dataset
        source_id = dataset_subset[i].get("source_dataset_id", None)
        
        # Extract data from result
        processed_data["prompt"].append(result["user_query"])
        processed_data["source_dataset_id"].append(source_id)
        processed_data["atomic_instructions"].append(result["atomic_instructions"].instructions)
        processed_data["reasoning"].append(result["reasoning"])
        processed_data["final_answer"].append(result["final_answer"])
        processed_data["num_iterations"].append(result["num_iterations"])
    
    # Create HuggingFace dataset
    results_dataset = Dataset.from_dict(processed_data)
    
    # Save as HuggingFace dataset
    print(f"Saving results to {output_path}...")
    results_dataset.save_to_disk(output_path)
    
    print("Processing complete!")
    print(f"Results saved to {output_path}")
    
    # Print brief summary
    print(f"\nProcessed {len(results_dataset)} queries")
    print(f"Dataset columns: {results_dataset.column_names}")
    
    # Calculate statistics
    instruction_counts = [len(instrs) for instrs in results_dataset["atomic_instructions"]]
    if instruction_counts:
        avg_instructions = sum(instruction_counts) / len(instruction_counts)
        print(f"Average instructions per example: {avg_instructions:.2f}")
        print(f"Max instructions in an example: {max(instruction_counts)}")
        print(f"Min instructions in an example: {min(instruction_counts)}")
    else:
        print("No instruction data available")
    
    # Push to Hugging Face Hub as a private dataset
    repo_id = "patrickfleith/reason-instruct-processed"
    print(f"\nPushing dataset to Hugging Face Hub as {repo_id}...")
    
    try:
        results_dataset.push_to_hub(
            repo_id=repo_id,
            private=True,  # Make the dataset private
            token=os.environ.get("HF_TOKEN"),  # Use token from environment
        )
        print(f"Dataset successfully pushed to {repo_id}!")
    except Exception as e:
        print(f"Error pushing to hub: {e}")
        print("Dataset is still saved locally at: {output_path}")
