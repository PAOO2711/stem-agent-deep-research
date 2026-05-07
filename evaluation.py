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
        The answer should demonstrate a clear structure, provide in-depth insights, and reference credible sources to support the claims made."""),
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