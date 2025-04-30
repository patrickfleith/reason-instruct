import os
import random
from dotenv import load_dotenv
from bespokelabs import curator
import prompts
from pydantic import BaseModel, Field
from typing import Tuple, List, Dict, Any
from datamodels import AtomicInstructions

# Load environment variables
load_dotenv()

# Set OpenAI API key if present in environment
if os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['CURATOR_DISABLE_CACHE'] = '1'


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
    
    llm = curator.LLM(model_name="gpt-4o-mini")
    analysis = llm(prompt).to_dict()
    return f"{analysis['response']}"


def generate_initial_answer(atomic_instructions: AtomicInstructions) -> Tuple[str, List[str]]:
    """Generate initial answer based on extracted atomic instructions.
    
    Args:
        atomic_instructions: The atomic instructions extracted from the user query
        
    Returns:
        initial_answer: str
    """
    # Deliberately select only a subset of instructions
    # Always include the first instruction, then each additional instruction with 50% probability
    import random
    
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

    llm = curator.LLM(model_name="gpt-4o-mini")
    initial_answer = llm(instructions_text).to_dict()
    return f"{initial_answer['response']}"


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
    
    llm = curator.LLM(model_name="gpt-4o-mini")
    refined_answer = llm(refinement_prompt).to_dict()
    return f"{refined_answer['response']}"


class InstructionAnalysis(BaseModel):
    explanation: str = Field(description="Explanation of why the instruction is satisfied or not")
    recommendation: str = Field(description="Recommendation to satisfy the instruction")
    satisfied: bool = Field(description="Whether the instruction is satisfied or not")


class InstructionVerificationResult(BaseModel):
    instruction: str = Field(description="The instruction that was verified")
    analysis: InstructionAnalysis = Field(description="Analysis of whether the instruction is satisfied")


class InstructionVerificationResults(BaseModel):
    results: List[InstructionVerificationResult] = Field(description="Verification results for each instruction")
    satisfied_count: int = Field(description="Number of instructions that are satisfied")
    total_count: int = Field(description="Total number of instructions")
    
    @property
    def satisfaction_ratio(self) -> float:
        """Calculate the ratio of satisfied instructions to total instructions"""
        return self.satisfied_count / self.total_count if self.total_count > 0 else 0.0


def verify_instruction(user_query: str, candidate_answer: str, instruction: str) -> InstructionAnalysis:
    """Verify a single instruction against a user query and candidate answer
    
    Args:
        user_query: The original user query
        candidate_answer: The answer to verify
        instruction: The specific instruction to verify
        
    Returns:
        Analysis of whether the instruction is satisfied
    """
    prompt = prompts.INSTRUCTION_VERIFICATION_TEMPLATE.format(
        user_query=user_query,
        candidate_answer=candidate_answer,
        instruction=instruction
    )
    
    # Use curator LLM for verification
    llm = curator.LLM(model_name="gpt-4o-mini", response_format=InstructionAnalysis)
    result = llm(prompt)
    
    # Extract values from the dataset response
    result_dict = result.to_dict()
    explanation = result_dict.get('explanation', ['No explanation provided'])[0]
    recommendation = result_dict.get('recommendation', ['No recommendation provided'])[0]
    satisfied = result_dict.get('satisfied', [False])[0]
    
    # Create and return an InstructionAnalysis object
    return InstructionAnalysis(
        explanation=explanation,
        recommendation=recommendation,
        satisfied=satisfied
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
) -> Tuple[list, str]:
    """Generate complete reasoning trace for a user query.
    
    The trace includes:
    1. Query analysis
    2. Initial answer
    3. Instruction verification and refinement
    
    Args:
        user_query: The user's query
        atomic_instructions: The atomic instructions extracted from the user query
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
    
    return reasoning_trace, current_answer


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


def process_query(user_query: str, atomic_instructions: AtomicInstructions, max_iterations: int = 3) -> Dict:
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
    """
    reasoning_trace, final_answer = generate_reasoning_trace(
        user_query=user_query,
        atomic_instructions=atomic_instructions,
        max_iterations=max_iterations
    )
    
    return {
        "user_query": user_query,
        "reasoning_trace": reasoning_trace,
        "reasoning_text": format_reasoning_trace(reasoning_trace),
        "final_answer": final_answer
    }


def process_dataset(queries: List[str], max_iterations: int = 3) -> List[Dict]:
    """Process a list of queries and return results for each.
    
    Args:
        queries: List of user queries to process
        max_iterations: Maximum number of refinement attempts per query
        
    Returns:
        List of result dictionaries, one per query
    """
    results = []
    
    for query in queries:
        result = process_query(query, max_iterations)
        results.append(result)
        
    return results


if __name__ == "__main__":
    from datasets import load_from_disk
    import json
    from pathlib import Path
    
    # Path to the processed dataset with instructions
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed_dataset_with_instructions")
    
    # Load the dataset
    print(f"Loading processed dataset from {dataset_path}...")
    try:
        dataset = load_from_disk(dataset_path)
        print(f"Dataset loaded successfully with {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Process the first 10 examples
    num_examples = 10
    first_examples = dataset.select(range(min(num_examples, len(dataset))))
    print(f"Processing first {len(first_examples)} examples...")
    
    # Store results
    results = []
    
    # Process each example
    for i, example in enumerate(first_examples):
        print(f"\n--- Processing example {i+1}/{len(first_examples)} ---")
        
        # Extract user query and atomic instructions
        user_query = example["prompt"]
        instructions_list = example["atomic_instructions"]
        
        # Create AtomicInstructions object
        atomic_instructions = AtomicInstructions(instructions=instructions_list)
        
        # Process the query
        print(f"Generating reasoning trace for: {user_query[:100]}...")
        result = process_query(
            user_query=user_query,
            atomic_instructions=atomic_instructions,
            max_iterations=3
        )
        
        results.append(result)
        print(f"Example {i+1} processed successfully")
    
    # Save results to a file
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "reasoning_traces.json")
    
    # Convert results to serializable format
    serializable_results = []
    for result in results:
        serializable_result = {
            "user_query": result["user_query"],
            "reasoning_trace": result["reasoning_trace"],
            "reasoning_text": result["reasoning_text"],
            "final_answer": result["final_answer"]
        }
        serializable_results.append(serializable_result)
    
    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Processed {len(results)} examples with reasoning traces")

