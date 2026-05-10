from model import llm_basic, llm_advanced
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class Reflection(BaseModel):
    revised_answer: str = Field(..., description="Improved, final answer text")
    changelog: List[str] = Field(default_factory=list, description="List of edits made")
    issues: List[str] = Field(default_factory=list, description="Problems identified in original")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence 0-1")


def reflect_fn(state: Dict[str, Any]) -> Dict[str, Any]:
    """Review and improve an existing answer.

    Returns a dict with keys `answer` (the revised answer string) and
    `reflection` (the parsed structured output as a dict). Falls back to
    raw model content if structured output fails.
    """
    answer = state.get("answer")
    if not answer:
        logger.warning("reflect_fn called without 'answer' in state")
        return {"answer": "", "reflection": {"error": "no answer provided"}}

    question = state.get("question", "")

    system_msg = (
        "You are an expert research reviewer. Review the provided answer and produce an improved,\n"
        "well-structured version. Follow the schema exactly."
    )

    user_msg = f"Question: {question}\nOriginal answer: {answer}\n\nReturn a JSON object matching the schema: revised_answer (string), changelog (list of short strings), issues (list of short strings), confidence (0-1)."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", user_msg)
    ])

    try:
        structured = llm_advanced.with_structured_output(
            Reflection,
            strict=True,
        ).invoke(prompt.format_messages())

        return {"answer": structured.revised_answer, "reflection": structured.dict()}
    except Exception as e:
        # Fallback: ask the model for a plain-text improvement
        logger.warning(f"reflect_fn structured output failed: {e}; falling back to plain text")
        fallback_prompt = f"Review and improve this answer, focusing on clarity and completeness.\nQuestion: {question}\nOriginal answer: {answer}\nProvide the improved answer only."
        response = llm_advanced.invoke(fallback_prompt).content
        return {"answer": response, "reflection": {"error": "structured_output_failed", "raw": response}}
