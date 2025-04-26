from datafast.llms import OpenAIProvider
from pydantic import BaseModel, Field
import prompts
from dotenv import load_dotenv
import os
from typing import List, Dict
from instruction_extractor import AtomicInstructions

# Load environment variables
load_dotenv()

# Initialize OpenAI provider
provider = OpenAIProvider(model_id="gpt-4.1")

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


# Example usage
if __name__ == "__main__":
    # Sample data
    user_query = "What does WISOP mean in Spacecraft Operations?"
    candidate_answer = "WISOP standards for World Interoperable Space Operations Protocol. It is often confused with other acronyms, such as ISO (International Organization for Standardization)."
    sample_instruction = "Need an answer to: What does WISOP mean in Spacecraft Operations?"
    
    # Test single instruction verification
    single_analysis = verify_instruction(
        user_query=user_query,
        candidate_answer=candidate_answer,
        instruction=sample_instruction
    )
    print("Single instruction verification:", single_analysis)
    
    # Test multiple instructions verification
    sample_atomic_instructions = AtomicInstructions(instructions=[
        "Provide the meaning of WISOP acronym",
        "Highlight common confusion",
    ])
    
    verification_results = verify_instructions(
        user_query=user_query,
        candidate_answer=candidate_answer,
        atomic_instructions=sample_atomic_instructions
    )
    
    print(f"\nMultiple instructions verification:\n")
    print(f"Satisfaction ratio: {verification_results.satisfaction_ratio:.2f} ({verification_results.satisfied_count}/{verification_results.total_count})")
    
    for i, result in enumerate(verification_results.results):
        print(f"\nInstruction {i+1}: {result.instruction}")
        print(f"Satisfied: {result.analysis.satisfied}")
        print(f"Explanation: {result.analysis.explanation}")
        if not result.analysis.satisfied:
            print(f"Recommendation: {result.analysis.recommendation}")

