from pydantic import BaseModel, Field
from model import llm_basic
from langchain_core.prompts import ChatPromptTemplate



class Evaluator(BaseModel):
    score: int = Field(default=0, description="Score from 0 to 10")
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
        - Depth: Does the answer provide in-depth insights and analysis? (0-2 points)
        - Structure: Is the answer well-organized and easy to follow? (0-2 points)
        - References: Are credible sources properly cited and used to support the claims? (0-2 points)

        In each category, a 0 indicates a poor performance, while a 2 indicates an excellent performance with no failures.

        Total score should be between 0 and 10.

        """),
        ("user", f"""
        Evaluate this answer from 0-10:
        - correctness
        - depth
        - structure
        - references

        Also give a detailed feedback on how to improve the answer based on the evaluation criteria.

        Question: {question}
        Answer: {answer}
        """)
    ])

    evaluation = llm_basic.with_structured_output(
        Evaluator,
        strict=True,
    ).invoke(prompt.format_messages())

    return evaluation.score, evaluation.feedback