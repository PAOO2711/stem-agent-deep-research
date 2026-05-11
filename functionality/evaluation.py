from pydantic import BaseModel, Field
from models.model import llm_advanced
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal

class Evaluator(BaseModel):
    #score: int = Field(default=0, description="Score from 0 to 10")
    correctness: int = Field(default=0, description="Correctness score (0-2)")
    depth: int = Field(default=0, description="Depth score (0-2)")
    structure: int = Field(default=0, description="Structure score (0-2)")
    sources: int = Field(default=0, description="Sources score (0-2)")
    bonus: int = Field(default=0, description="Bonus points (0-2)")
    penalties: int = Field(default=0, description="Total penalties deducted")
    feedback: str = Field(default="", description="Feedback for improvement")


class FeedbackSignals(BaseModel):
    missing_information: bool = Field(default=False, description="Answer is missing important information")
    needs_multi_step_reasoning: bool = Field(default=False, description="Question needs a multi-step reasoning approach")
    poor_structure: bool = Field(default=False, description="Answer structure is weak or hard to follow")
    research_gaps: bool = Field(default=False, description="There are gaps in research coverage or sources")
    weak_sources: bool = Field(default=False, description="Sources are missing, weak, or not credible enough")
    hallucinations: bool = Field(default=False, description="Answer contains hallucinated or unsupported claims")
    severity: Literal["low", "medium", "high"] = Field(default="low", description="Overall severity of the issues")
    feedback_text: str = Field(default="", description="Short human-readable summary of the issues")

def evaluate(question: str, answer: str) -> (float, FeedbackSignals):
    """Evaluate the agent's answer and return a score and feedback."""
    
    system_message = """You are an expert evaluator for deep research questions.
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
    Scores above 8 are rare and only for near-perfect answers."""

    user_prompt = f"""Evaluate this answer from 0-10:
    - correctness
    - depth
    - structure
    - sources
    - penalties
    - bonus

    Also return structured feedback signals that can be used by the mutation step.

    Question: {question}
    Answer: {answer}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", user_prompt)
    ])

    evaluation = llm_advanced.with_structured_output(
        Evaluator,
        strict=True,
    ).invoke(prompt.format_messages(question=question, answer=answer))

    feedback_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict classifier that converts an evaluation into structured feedback signals.
Return only the schema fields and no extra text."""),
        ("user", f"""Question: {question}
Answer: {answer}

Evaluation scores:
correctness={evaluation.correctness}
depth={evaluation.depth}
structure={evaluation.structure}
sources={evaluation.sources}
bonus={evaluation.bonus}
penalties={evaluation.penalties}
feedback={evaluation.feedback}

Classify the answer into structured signals with severity.""")
    ])

    feedback = llm_advanced.with_structured_output(
        FeedbackSignals,
        strict=True,
    ).invoke(feedback_prompt.format_messages())

    total_score = evaluation.correctness + evaluation.depth + evaluation.structure + evaluation.sources + evaluation.bonus - abs(evaluation.penalties)
    total_score = max(0, min(10, total_score))  # Clamp between 0 and 10

    return total_score, evaluation, feedback

def stop_condition(score: int) -> bool:
    """Determine if the score is good enough to stop further iterations."""
    return score > 8  # Threshold for stopping