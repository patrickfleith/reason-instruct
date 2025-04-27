from datafast.llms import OpenAIProvider
import prompts
from dotenv import load_dotenv
import os
from typing import List, Dict, Tuple
from datamodels import AtomicInstructions, InstructionAnalysis, InstructionVerificationResult, InstructionVerificationResults
from instruction_verifier import generate_critique, verify_instructions, verify_instruction

# Load environment variables
load_dotenv()

# Initialize OpenAI providers
query_analyzer_provider = OpenAIProvider(model_id="gpt-4.1-mini")
answer_provider = OpenAIProvider(model_id="gpt-4.1-mini")
refinement_provider = OpenAIProvider(model_id="gpt-4.1")
instruction_extractor_provider = OpenAIProvider(model_id="gpt-4.1-mini")
verification_provider = OpenAIProvider(model_id="gpt-4.1-mini")


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
    return f"Query Analysis:\n{analysis}"


def extract_instructions(user_query: str) -> AtomicInstructions:
    """Extract atomic instructions from the user query.
    
    Args:
        user_query: The query from the user
        
    Returns:
        AtomicInstructions object containing extracted instructions
    """
    prompt = prompts.INSTRUCTION_EXTRACTION_TEMPLATE.format(
        user_prompt=user_query
    )
    
    instructions = instruction_extractor_provider.generate(
        prompt=prompt,
        response_format=AtomicInstructions
    )
    
    return instructions


def generate_initial_answer(user_query: str) -> str:
    """Generate initial answer to user query.
    
    Args:
        user_query: The query from the user
        
    Returns:
        Initial answer to the query
    """
    prompt = f"Please answer this question or follow this instruction: {user_query}"
    
    answer = answer_provider.generate(prompt=prompt)
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
    
    refined_answer = answer_provider.generate(prompt=refinement_prompt)
    return refined_answer


def generate_reasoning_trace(
    user_query: str,
    max_iterations: int = 3
) -> Tuple[list, str]:
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
    
    # Step 2: Extract instructions
    atomic_instructions = extract_instructions(user_query)
    # instructions_text = "Extracted instructions:\n" + "\n".join([f"- {instr}" for instr in atomic_instructions.instructions])
    # reasoning_trace.append(instructions_text)
    
    # Step 3: Generate initial answer
    initial_answer = generate_initial_answer(user_query)
    reasoning_trace.append(f"I'll draft an initial answer:\n{initial_answer}")
    
    # Step 4: Verify and refine
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


def process_query(user_query: str, max_iterations: int = 3) -> Dict:
    """Process a single user query and return results.
    
    Args:
        user_query: The user's query
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


# Example usage
if __name__ == "__main__":
    example_query = "Create a web page that displays a list of products. Each product should have an image, name, price, and 'Add to Cart' button. When a user clicks on the button, the product should be added to a shopping cart."
    
    # Process a single query
    result = process_query(example_query)
    
    # Print the reasoning trace and final answer
    print(f"=== REASONING TRACE ===\n")
    print(result["reasoning_text"])
    
    print("\n" + "=" * 50)
    print("\nFINAL ANSWER:\n")
    print(result["final_answer"])
    
    # Example of processing multiple queries
    """
    example_queries = [
        "How does climate change affect biodiversity? What are its effects on seas ecosystems? Focus on Mediterranean Sea",
        "What are the benefits of regular exercise? explain what is best to loose weight: exercise or food intake reduction",
        "Explain the concept of machine learning to a 10-year-old and use analogies."
    ]
    
    results = process_dataset(example_queries)
    
    for i, result in enumerate(results):
        print(f"\n\n=== QUERY {i+1} ===")
        print(f"Query: {result['user_query']}")
        print(f"Answer: {result['final_answer']}")
    """
