from datafast.llms import OpenAIProvider
import prompts
from dotenv import load_dotenv
from datamodels import AtomicInstructions, InstructionAnalysis, InstructionVerificationResult, InstructionVerificationResults
from instruction_verifier import generate_critique
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

# Initialize OpenAI provider
provider = OpenAIProvider(model_id="gpt-4.1")


def colorize_critique(critique: str) -> str:
    """Add color formatting to the critique string for better terminal readability.
    
    Args:
        critique: The original critique string without color
        
    Returns:
        A formatted critique string with terminal colors
    """
    import re
    
    # Colorize only the most important elements
    colored = critique
    
    # Make the header stand out
    colored = colored.replace("Instruction Verification Results", f"{Fore.CYAN}{Style.BRIGHT}Instruction Verification Results{Style.RESET_ALL}")
    
    # Colorize status indicators - these are the most important visual cues
    colored = colored.replace("✓ Satisfied", f"{Fore.GREEN}✓ Satisfied{Style.RESET_ALL}")
    colored = colored.replace("✗ Not satisfied", f"{Fore.RED}✗ Not satisfied{Style.RESET_ALL}")
    
    return colored


def verify_instruction(user_query: str, candidate_answer: str, instruction: str) -> InstructionAnalysis:
    """Verify a single instruction against a user query and candidate answer"""
    prompt = prompts.INSTRUCTION_VERIFICATION_TEMPLATE.format(
        user_query=user_query,
        candidate_answer=candidate_answer,
        instruction=instruction
    )
    
    return provider.generate(
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

def generate_reasoning_trace(
    user_query: str,
    candidate_answer: str,
    atomic_instructions: AtomicInstructions,
    max_iterations: int = 3
) -> (list, str):
    """Generate a reasoning trace by iteratively refining an answer until all instructions are satisfied.
    
    Args:
        user_query: The original user query
        candidate_answer: The initial answer to verify
        atomic_instructions: The set of instructions to verify against
        max_iterations: Maximum number of refinement attempts
        
    Returns:
        A tuple containing (reasoning_trace, final_answer) where reasoning_trace is a list of strings
        representing the reasoning process and final_answer is the best answer produced
    """
    # Initial answer to refine
    current_answer = candidate_answer
    iteration = 0
    all_satisfied = False
    reasoning_trace = []

    while iteration < max_iterations:
        # Increment iteration counter first
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
        refinement_prompt = prompts.ANSWER_REFINEMENT_TEMPLATE.format(
            user_query=user_query,
            candidate_answer=current_answer,
            critique=critique
        )
        
        refined_answer = provider.generate(
            prompt=refinement_prompt
        )
        
        # Add the refined answer to trace and update for next iteration
        reasoning_trace.append(f"Refined Answer:\n{refined_answer}")
        current_answer = refined_answer

    # Add final assessment message to trace
    if not all_satisfied:
        reasoning_trace.append(f"\nAfter {iteration} refinement attempts, {verification_results.total_count - verification_results.satisfied_count} "  
                     f"instruction(s) still need attention. Using the best available answer.")
    
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


# Example usage
if __name__ == "__main__":
    # Sample data
    user_query = "How does climate change affect biodiversity? Can you also explain the impact on ocean ecosystems?"
    candidate_answer = "Climate change is bad! Not cool!"

    sample_atomic_instructions = AtomicInstructions(instructions=[
        "Explain how climate change affects biodiversity",
        "Highlight the impact of climate change on ocean ecosystems",
    ])

    print(f"{Fore.CYAN}Starting verification process...{Style.RESET_ALL}\n")
    
    # Generate reasoning trace and get final answer
    reasoning_trace, final_answer = generate_reasoning_trace(
        user_query=user_query,
        candidate_answer=candidate_answer,
        atomic_instructions=sample_atomic_instructions,
        max_iterations=3
    )
    
    # Print the reasoning trace as separate items with colors where appropriate
    print("\n=== TRACE AS SEPARATE ITEMS ===")
    for i, item in enumerate(reasoning_trace):
        if i > 0:
            print("\n" + "-" * 30)
            
        if item.startswith("Let's verify that the answer satisfies all instructions:"):
            print(colorize_critique(item))
        else:
            print(item)
    
    # Format and print the reasoning trace as a single string
    print("\n\n=== TRACE AS A SINGLE STRING ===")
    formatted_trace = format_reasoning_trace(reasoning_trace)
    print(formatted_trace)
    
    # Print final answer
    print("\n" + "=" * 50)
    print("\nFINAL ANSWER:\n")
    print(final_answer)
    
        
