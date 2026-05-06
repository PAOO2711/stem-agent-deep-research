from pydantic import BaseModel, Field
from model import llm_basic
from langchain_core.prompts import ChatPromptTemplate



class Evaluator(BaseModel):
    score: int = Field(default=0, description="Score from 0 to 10"),
    feedback: str = Field(default="", description="Feedback for improvement")

def evaluate(question: str, answer: str) -> (float, str):
    """Evaluate the agent's answer and return a score and feedback."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert evaluator for deep research questions."),
        ("user", f"""
        Evaluate this answer from 1-10:
        - correctness
        - depth
        - structure
        - references (if applicable)
        
        Also give a detailed feedback on how to improve the answer based on the evaluation criteria.

        Question: {question}
        Answer: {answer}
        """)
    ])

    evaluation = llm_basic.with_structured_output(
        Evaluator,
        strict=True,
    ).invoke(prompt)

    return evaluation.score, evaluation.feedback