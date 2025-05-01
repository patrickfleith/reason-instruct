from pydantic import BaseModel, Field

# Define data models
class AtomicInstructions(BaseModel):
    """Container for atomic instructions extracted from user query."""
    instructions: list[str] = Field(description="List of atomic instructions extracted from the user query.")

class InstructionAnalysis(BaseModel):
    explanation: str = Field(description="Explanation of why the instruction is satisfied or not")
    recommendation: str = Field(description="Recommendation to satisfy the instruction")
    satisfied: bool = Field(description="Whether the instruction is satisfied or not")

class InstructionVerificationResult(BaseModel):
    instruction: str = Field(description="The instruction that was verified")
    analysis: InstructionAnalysis = Field(description="Analysis of whether the instruction is satisfied")

class InstructionVerificationResults(BaseModel):
    results: list[InstructionVerificationResult] = Field(description="Verification results for each instruction")
    satisfied_count: int = Field(description="Number of instructions that are satisfied")
    total_count: int = Field(description="Total number of instructions")
    
    @property
    def satisfaction_ratio(self) -> float:
        """Calculate the ratio of satisfied instructions to total instructions"""
        return self.satisfied_count / self.total_count if self.total_count > 0 else 0.0
