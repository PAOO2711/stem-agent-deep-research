from pydantic import BaseModel, Field
from model import llm_advanced
from langchain_core.prompts import ChatPromptTemplate

class Evaluator(BaseModel):
    #score: int = Field(default=0, description="Score from 0 to 10")
    correctness: int = Field(default=0, description="Correctness score (0-2)")
    depth: int = Field(default=0, description="Depth score (0-2)")
    structure: int = Field(default=0, description="Structure score (0-2)")
    sources: int = Field(default=0, description="Sources score (0-2)")
    bonus: int = Field(default=0, description="Bonus points (0-2)")
    penalties: int = Field(default=0, description="Total penalties deducted")
    feedback: str = Field(default="", description="Feedback for improvement")

def evaluate(question: str, answer: str) -> (float, str):
    """Evaluate the agent's answer and return a score and feedback."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert evaluator for deep research questions.
        A deep research question requires comprehensive understanding, critical analysis, and synthesis of information from multiple sources. 
        The answer should demonstrate a clear structure, provide in-depth insights, and reference credible sources to support the claims made.
        If the question should be searched on articles or papers, ensure the answer includes relevant citations.
        
        How to evaluate:
        - Correctness: Is the answer factually accurate and does it address the question directly? (0-2 points)
            0: incorrect or irrelevant answer
            1: partially correct but incomplete
            2: fully correct and directly answers question
        - Depth: Does the answer provide in-depth insights and analysis? (0-2 points)
            0: superficial or single sentence
            1: some explanation but shallow
            2: deep reasoning, multiple aspects covered
        - Structure: Is the answer well-organized and easy to follow? (0-2 points)
            0: disorganized or hard to follow
            1: somewhat structured 
            2: clear sections, logical flow 
        - Sources: Are credible sources properly cited and used to support the claims? (0-2 points)
            0: no sources cited
            1: sources cited but low quality or generic
            2: high-quality sources (papers, docs, reputable articles) properly cited
        - Penalties. Deduct points if:
            - no citations when required (-2)
            - each hallucinated claim: (-1)
            - severe hallucination: (-2)
            - shallow reasoning (-1 to -2)
        
        BONUS (0–2 points):
        - 0: basic answer, no synthesis
        - 1: strong reasoning or good synthesis of ideas
        - 2: publication-level answer:
            - integrates multiple perspectives
            - includes nuanced reasoning
            - anticipates counterarguments
            - demonstrates expert-level synthesis

        

        Be strict. Most answers should score between 3 and 7.
        Scores above 8 are rare and only for near-perfect answers.

        """),
        ("user", f"""
        Evaluate this answer from 0-10:
        - correctness
        - depth
        - structure
        - sources
        - penalties
        - bonus

        Also give a detailed feedback showing the evaluation phase and how to improve the answer based on the evaluation criteria.

        Question: {question}
        Answer: {answer}
        """)
    ])

    evaluation = llm_advanced.with_structured_output(
        Evaluator,
        strict=True,
    ).invoke(prompt.format_messages())

    total_score = evaluation.correctness + evaluation.depth + evaluation.structure + evaluation.sources + evaluation.bonus - evaluation.penalties
    total_score = max(0, min(10, total_score))  # Clamp between 0 and 10

    return total_score, evaluation.feedback

def stop_condition(score: int) -> bool:
    """Determine if the score is good enough to stop further iterations."""
    return score > 8  # Threshold for stopping