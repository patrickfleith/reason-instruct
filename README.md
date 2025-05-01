# 🤓 Instruction-freak-reasoning

## 🎯 Purpose and Scope

Have you ever felt frustrated when a language model does not follow some of the instructions you provided? 

**Instruction-freak-reasoning** is a dataset designed to fix that issue. Given a user query, the reasoning trace:

1. Breaks down the instructions into atomic components
2. Makes an initial answer (possibly incomplete)
3. Systematically verifies the candidate answer against each atomic instruction
4. Iteratively refines the answer until all atomic instructions are satisfied

This dataset uses a mixture of curated datasets to start with realistic user queries. The reasoning traces are generated using multiple non-reasoning LLM calls working in concert.

## 💻 How the Dataset Was Created

### 💡 Key Concept: Atomic Instructions

Atomic instructions are the fundamental building blocks of complex queries, extracted using **Bespokelab's Curator** framework.  
Each atomic instruction represents a single, indivisible requirement that can be independently verified.

This approach enables:

1. Precise evaluation of response completeness by tracking which instructions are satisfied
2. Structured refinement cycles based on instruction-specific feedback
3. Quantitative assessment of reasoning quality through satisfaction ratios

The atomicity of instructions enables us to start from non-reasoning models, generating `instruction-freak-reasoning` traces which can be further used for fine-tuning reasoning models.

### 👇 Preprocessing

1. Started with 2,500 samples from each of the following source datasets:
   - `argilla/ifeval-like-data`
   - `allenai/tulu-3-sft-personas-instruction-following`
   - `HuggingFaceH4/ultrafeedback_binarized`
   - `HuggingFaceH4/no_robots`
2. Filtered out prompts that were too short or too long
3. Applied semantic deduplication with Semash (aggressive threshold of 0.5)

> This heavy filtering reduced the size of the dataset to 1,500 samples (**-85%**)

4. Used `bespokelab-curator` to extract atomic instructions from the user queries

### 🔬 Reasoning Traces Generation

1. **Query analysis**: Using Claude 3.5 Haiku to understand user intent and context
2. **Deliberate partial implementation**: Initial answers intentionally address only a subset of atomic instructions (first instruction always included, others with 50% probability)
3. **Systematic verification**: Each atomic instruction is independently verified against the answer using GPT-4.1
4. **Targeted refinement**: Answers are iteratively improved based on verification feedback until all instructions are satisfied or max iterations reached
5. **Multiple LLM collaboration**: Process leverages different models for different steps (Claude for drafting, GPT-4.1 for verification and refinement)

Finally, we save the complete `reasoning` traces and the `final_answer`.

> **Note**: We only generated reasoning traces for 200 examples due to the cost of the process. Scaling could easily be expanded.

### 🔥 Post-processing

After reasoning trace generation, I manually inspected several examples and noticed some final answers were error messages or not fully valid answers.

- Used `bespokelabs-curator` to validate the `final_answer` against the initial user prompt
- Dropped invalid answers (179 instances remained after filtering)

## 📚 Dataset Composition and Structure

Each example in the dataset contains:

- **prompt**: The original user prompt/query
- **source_dataset_id**: Origin of the original query  
- **atomic_instructions**: Explicit list of atomic instructions extracted from the query
- **reasoning**: Complete trace of the reasoning process, including:
  - Query analysis
  - Initial answer generation
  - Instruction verification results
  - Critiques of unsatisfied instructions
  - Refinement iterations
- ✨ **final_answer**: The best answer after refinement attempts
- 📋 **num_iterations**: How many candidate attempts were generated

## 🧠 Usage for Fine-tuning

For fine-tuning applications, the dataset would benefit from scaling to a larger size:

- ➕ More starter datasets
- ➕ More reasoning traces generations

### Resources for Fine-tuning 
- 🤗 [HuggingFace LLM Course](https://huggingface.co/learn/llm-course/en/chapter12/5?fw=pt)
- 🚀 [Unsloth Documentation](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl)

## ⚠️ Limitations

- The atomic instruction extraction depends on the quality of the LLM (GPT-4.1) used
- The current version contains a limited number of instruction types (<200) (but can be scaled)
- English-only dataset
- Responses and verifications are fully generated - potential risk for hallucinations

## 📃 Code Repository

The code is available [here](https://github.com/patrickfleith/reason-instruct).

## 💻 Citation

If you use this dataset in your research, please cite it as:

```bibtex
@dataset{instruction-freak-reasoning-2025,
  title = {Instruction Freak Reasoning: A Dataset for Atomic Instruction Following and Verification},
  author = {Patrick Fleith},
  year = {2025},
  url = {https://huggingface.co/datasets/patrickfleith/instruction-freak-reasoning}
}
```

## ⚖️ License

[Creative Commons Attribution Non Commercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

The prompts belong to the original datasets and are subject to their respective licenses:

- 📂 `argilla/ifeval-like-data`: [Qwen License](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE)
- 📂 `allenai/tulu-3-sft-personas-instruction-following`: Open Data Commons License Attribution family
- 📂 `HuggingFaceH4/ultrafeedback_binarized`: MIT license
- 📂 `HuggingFaceH4/no_robots`: Creative Commons Attribution Non Commercial 4.0
