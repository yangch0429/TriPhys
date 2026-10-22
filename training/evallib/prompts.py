"""Prompt定义 - System + User分离"""

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


def build_user_prompt(question: str, question_type: str) -> str:
    """构建用户prompt，根据题型添加格式要求"""
    
    prompt = f"""Question: {question}

"""
    
    # 根据题型添加格式说明
    if question_type == "TYPE_A":
        prompt += """Answer format: Provide a numerical value with appropriate unit (e.g., "10.5 J", "3.2 m/s").

"""
    elif question_type == "TYPE_B":
        prompt += """Answer format: Provide either a single value OR a range (e.g., "5 J" or "4-6 J").

"""
    elif question_type == "TYPE_C":
        prompt += """Answer format: Start with "Yes" or "No", followed by quantitative justification (e.g., "Yes, because the kinetic energy is 10 J which exceeds the threshold").

"""
    elif question_type == "TYPE_D":
        prompt += """Answer format: Specify the comparison direction and magnitude (e.g., "Object A has 2x greater kinetic energy than Object B").

"""
    
    return prompt
