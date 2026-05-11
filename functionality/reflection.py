from models.model import llm_basic, llm_advanced
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


def split_sources_section(answer: str) -> tuple[str, str]:
    """Split an answer into body text and a trailing Sources section."""
    if "\nSources:\n" not in answer:
        return answer, ""

    body, sources = answer.split("\nSources:\n", 1)
    return body, sources


def reflect_fn(state: Dict[str, Any]) -> Dict[str, Any]:
    """Review and improve an existing answer while preserving sources.

    Extracts any 'Sources:' section from the original answer before reflection,
    reflects on the main content, then re-appends the sources at the end.
    
    Returns a dict with keys `answer` (the revised answer string) and
    `reflection` (the parsed structured output as a dict). Falls back to
    raw model content if structured output fails.
    """

    #logger.info("Starting reflection phase to review and improve the answer")
    answer = state.get("refined_answer") or state.get("answer")
    if not answer:
        logger.warning("reflect_fn called without 'answer' in state")
        return {"answer": "", "reflection": {"error": "no answer provided"}}

    question = state.get("question", "")

    # Extract and preserve sources section if present
    answer_without_sources, sources_block = split_sources_section(answer)
    sources_section = f"\nSources:\n{sources_block}" if sources_block else ""

    system_msg = (
        "You are an expert research reviewer. Review the provided answer and produce an improved,\n"
        "well-structured version. Follow the schema exactly."
    )

    user_msg = f"Question: {question}\nOriginal answer: {answer_without_sources}\n\nReturn a JSON object matching the schema: revised_answer (string), changelog (list of short strings), issues (list of short strings), confidence (0-1)."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", user_msg)
    ])

    try:
        structured = llm_advanced.with_structured_output(
            Reflection,
            strict=True,
        ).invoke(prompt.format_messages())

        # Re-append sources section if it existed
        revised_answer, _ = split_sources_section(structured.revised_answer)
        if sources_section:
            revised_answer = revised_answer + sources_section

        #logger.info("Reflection complete: Answer improved with structured feedback")

        return {"answer": revised_answer, "reflection": structured.dict()}
    except Exception as e:
        # Fallback: ask the model for a plain-text improvement
        logger.warning(f"reflect_fn structured output failed: {e}; falling back to plain text")
        fallback_prompt = f"Review and improve this answer, focusing on clarity and completeness.\nQuestion: {question}\nOriginal answer: {answer_without_sources}\nProvide the improved answer only."
        response = llm_advanced.invoke(fallback_prompt).content
        response, _ = split_sources_section(response)
        
        # Re-append sources section if it existed
        if sources_section:
            response = response + sources_section
        
        return {"answer": response, "reflection": {"error": "structured_output_failed", "raw": response}}

