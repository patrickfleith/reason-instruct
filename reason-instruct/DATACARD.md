# Reason-Instruct Dataset

## Dataset Description

Have you ever felt frustrated when a language model does not follow some the instructions you provided? 

Reason-Instruct is a dataset designed to fix that issue by reasoning language models to closely follow instructions. Given a user query, the reasoning trace:
1. breaks down the instructions into atomic components
2. makes an initial answer
3. systematically verifies each atomic instruction
4. iteratively refines the answer until all atomic instructions are satisfied

## Example:

**User query:**
> *"Can you provide a brief summary of the benefits of using solar energy, including its environmental impact and cost-effectiveness? Your response should contain at least 3 sentences and highlight at least 2 sections using markdown. *solar power* should appear at least 2 times."*

**Reasoning trace example:**

#### Step 1: Query Analysis
> The user wants a concise summary explaining the advantages of *solar power*. The summary needs to cover:
>
> 1.  **Environmental Impact:** How does *solar power* benefit the environment?
> 2.  **Cost-Effectiveness:** Is *solar power* a financially sound choice?
>
> **Specific Instructions:**
>
> *   **Length:** The summary should be at least 3 sentences long.
> *   **Markdown:** At least two sections of the response should be highlighted using markdown formatting (e.g., bold, italics).
> *   **Keyword Repetition:** The phrase "*solar power*" must appear at least twice in the response.

#### Step 2: Initial Answer Generation
> Here's a brief summary of the benefits of using solar energy:
>
> Solar energy offers numerous advantages, making it a compelling alternative to traditional energy sources.
>
> *Environmental Benefits:*
> Solar power significantly reduces reliance on fossil fuels, leading to lower greenhouse gas emissions and mitigating climate change. Unlike coal and natural gas, solar energy doesn't release harmful pollutants into the air, improving air quality and reducing health problems associated with air pollution. Furthermore, it minimizes water usage compared to conventional power plants, which require substantial amounts of water for cooling.
>
> *Cost-Effectiveness:*
> While the initial investment in solar panels can be significant, the long-term operational costs are minimal. Once installed, solar power provides free energy from the sun, substantially reducing or eliminating electricity bills. Government incentives, such as tax credits and rebates, can further reduce the initial investment. Over the lifespan of a solar system, the savings can outweigh the initial costs, making it a cost-effective energy solution. Furthermore, generating your own energy through solar panels increases the independence of the homeowner.

#### Step 3: Instruction Verification
> **Instruction 1:** Provide a brief summary of the benefits of using solar energy, including its environmental impact and cost-effectiveness.
> - **Explanation:** The candidate answer provides a brief summary focused on the benefits of using solar energy, and it discusses both the environmental impact (e.g., reduction in emissions, air quality improvement) and cost-effectiveness (e.g., reduced electricity bills, incentives, long-term savings). All aspects of the instruction are covered.
> - **Status:** ✓ Satisfied
>
> **Instruction 2:** Ensure your response contains at least 3 sentences.
> - **Explanation:** The candidate answer contains multiple sentences, with clearly more than three sentences present in the response. Both the environmental and cost-effectiveness sections contain several sentences each.
> - **Status:** ✓ Satisfied
>
> **Instruction 3:** Highlight at least 2 sections using markdown.
> - **Explanation:** The candidate answer uses markdown to highlight two sections: '*Environmental Benefits:*' and '*Cost-Effectiveness:*', which are clearly separated with markdown-style headings.
> - **Status:** ✓ Satisfied
>
> **Instruction 4:** Include the phrase 'solar power' at least 2 times.
> - **Explanation:** The candidate answer includes the phrase 'solar power' exactly two times: once in the Environmental Benefits section ('Solar power significantly reduces reliance on fossil fuels...') and once in the Cost-Effectiveness section ('solar power provides free energy from the sun...'). This meets the requirement of the instruction to include the phrase at least 2 times.
> - **Status:** ✓ Satisfied
>
> **Overall assessment:** All instructions have been satisfied. I can now answer the user query.

If the answer would not be satifying all intruction, a refinement process is performed until either the instructions are satisfied.

After the reasoning trace, the model provides the final answer.

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
