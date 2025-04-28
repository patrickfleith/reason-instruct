# Reason-Instruct Dataset

## Dataset Description

Reason-Instruct is a dataset designed to improve how reasoning language models follow complex instructions by breaking them down into atomic components, systematically verifying each step, and iteratively refining the answer until all instructions are satisfied

## Motivation

Many existing instruction datasets focus on single-turn responses without providing insight into the reasoning process or explicit verification of whether all parts of an instruction were followed. Reason-Instruct addresses this gap by:

1. Explicitly extracting atomic instructions from complex user queries
2. Verifying instruction satisfaction through structured evaluation
3. Capturing the iterative refinement process when instructions are missed until all instructions are satisfied

## Composition and Structure

Each example in the dataset contains:

- **user_query**: The original user prompt/query
- **atomic_instructions**: Explicit list of atomic instructions extracted from the query
- **reasoning_trace**: Complete trace of the reasoning process, including:
  - Query analysis
  - Initial answer generation
  - Instruction verification results
  - Critiques of unsatisfied instructions
  - Refinement attempts
- **reasoning_text**: Formatted version of the reasoning trace
- **final_answer**: The best answer after refinement attempts
- **source_dataset**: Origin of the original query
- **num_refinements**: Number of refinement iterations performed
- **all_instructions_satisfied**: Boolean indicating whether all instructions were satisfied by the final answer

## Collection Process

The dataset was created through a structured workflow:

1. Source queries were collected from a diverse range of existing datasets
2. LLM-based instruction extractors (using GPT-4.1) identified atomic instructions within each query
3. Initial answers were deliberately generated to satisfy only a subset of instructions
4. Instruction verifiers evaluated each atomic instruction against the answer
5. Critiques were generated for unsatisfied instructions
6. Answers were refined up to 3 times, with verification after each refinement
7. The entire process was recorded to create rich reasoning traces

## Intended Uses

This dataset is designed for:

- Training and evaluating language models on complex instruction following
- Improving reasoning about atomic instructions in user queries
- Benchmarking the effectiveness of instruction verification systems
- Studying the refinement process in language models

## Limitations

- The atomic instruction extraction depends on the quality of the LLM (GPT-4.1) used
- Verification judgments are subjective and may contain biases from the verification model
- The dataset may not cover all possible forms of instructions or reasoning patterns
- The refinement process is limited to 3 iterations, which may not be sufficient for all cases

## Citation and Acknowledgments

If you use this dataset in your research, please cite it as:

```
@dataset{reason-instruct-2025,
  title = {Reason-Instruct: A Dataset for Atomic Instruction Following and Verification},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/reasoning-datasets-competition}
}
```

## License

[Insert appropriate license information here]
