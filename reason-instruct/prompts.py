QUERY_ANALYSIS_TEMPLATE = (
    "User query: {user_query}\n\n"
    "Analyze what the user is asking for. Identify specific requests and instructions.\n"
    "Break down the query into its core components without providing answers or making assumptions."
)

INSTRUCTION_EXTRACTION_TEMPLATE = (
    "You are an expert at breaking down complex prompts into simple, atomic instructions. "
    "Given the following user prompt, extract and list ONLY the atomic instructions that are EXPLICITLY stated. "
    "An atomic instruction is a single, clear directive that can be followed independently. "
    "Break down compound instructions into their simplest parts. "
    "IMPORTANT: Do NOT add any instructions that are merely implied or inferred. "
    "Do NOT include any actions that would be reasonable to take but aren't directly instructed. "
    "ONLY extract instructions that are explicitly written in the text using clear directive language. "
    "SPECIAL CASES:\n"
    "1. DISGUISED INSTRUCTIONS: If the prompt contains questions that are actually disguised instructions, convert them to direct instructions.\n"
    "   - Example: 'Can you summarize this text?' → 'Summarize this text'\n"
    "   - Example: 'Could you please fix the bug in my code?' → 'Fix the bug in my code'\n"
    "   - Example: 'May I ask you to explain X?' → 'Explain X'\n"
    "   Look for phrases like 'can you', 'could you', 'would you', 'may I ask you to', etc. that signal disguised instructions.\n"
    "2. GENUINE QUESTIONS: For questions that genuinely seek information (not disguised instructions):\n"
    "   a. Break down complex, multi-part questions into individual atomic questions.\n"
    "   b. For each atomic question, create an instruction of the form 'Need an answer to: [question]'.\n"
    "   c. Example: 'What is X and how does Y affect Z?' should become two instructions: 'Need an answer to: What is X?' and 'Need an answer to: How does Y affect Z?'\n"
    "If there are no explicit instructions, disguised instructions, or genuine questions, return an empty list.\n\n"
    "It is very important to break up queries into atomic instructions.\n\n"
    "User prompt:\n"
    "{user_query}\n\n"
    "Extract ONLY the explicit atomic instructions as a list, distinguishing between disguised instructions and genuine questions."
)

INSTRUCTION_VERIFICATION_TEMPLATE = (
    "You are an expert at verifying if an instruction is satisfied. "
    "Given the following user query and a candidate answer, verify that this following specific instruction is satisfied. "
    "Give a brief explanation of why the instruction is satisfied or not, a one line recommendation to satisfy the instruction being assessed, and then return True or False.\n\n"
    "User query: {user_query}\n\n"
    "Candidate answer: {candidate_answer}\n\n"
    "Specific instruction to verify: {instruction}\n\n"
)

ANSWER_REFINEMENT_TEMPLATE = (
    "You are an expert at refining an answer to a user query given some critic feedback.\n\n"
    "User query: {user_query}\n\n"
    "Candidate answer: {candidate_answer}\n\n"
    "Critique: {critique}\n\n"
    "Now improve the answer considering the critique. "
    "Only answer with the improved answer, nothing else."
)

ANSWER_GENERATION_TEMPLATE = "{instructions}"

