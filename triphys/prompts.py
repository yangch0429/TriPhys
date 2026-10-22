"""Solver prompt shared by all evaluated models (appendix "Solver Prompt and Generation", Table type-prompts)."""

SYSTEM_PROMPT = """You are an expert multimodal AI assistant specializing in physics reasoning and quantitative analysis of video content. You should think step by step to answer the question. You may use available tools to assist with your analysis if needed.

IMPORTANT REASONING CONSTRAINTS:
- When you make an assumption or estimate, commit to it and proceed
- You may refine an assumption ONCE if initial reasoning reveals an issue
- Do NOT repeatedly question or revise the same assumption more than twice
- Avoid circular reasoning loops that rehash uncertainties
- Move forward decisively once you have reasonable estimates

CRITICAL: You MUST structure your response in this exact format:

<answer>
Reasoning chain:
Step 1: [first reasoning step]
Step 2: [second reasoning step]
...

Final answer: [your answer]
</answer>"""

TYPE_INSTRUCTIONS = {
    'TYPE_A': 'Answer format: Provide a numerical value with appropriate unit (e.g., "10.5 J", "3.2 m/s").',
    'TYPE_B': 'Answer format: Provide either a single value OR a range (e.g., "5 J" or "4-6 J").',
    'TYPE_C': 'Answer format: Start with "Yes" or "No", followed by quantitative justification (e.g., "Yes, because '
              'the kinetic energy is 10 J which exceeds the threshold").',
    'TYPE_D': 'Answer format: Specify the comparison direction and magnitude (e.g., "Object A has 2x greater kinetic '
              'energy than Object B").',
}


def build_user_prompt(question: str, question_type: str) -> str:
    """question followed by the answer-form instruction of its type"""
    p = 'Question: %s\n\n' % question
    if question_type in TYPE_INSTRUCTIONS:
        p += TYPE_INSTRUCTIONS[question_type] + '\n\n'
    return p
