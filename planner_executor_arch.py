"""
Planner-Executor Architecture for Deep Research
================================================
This module implements a sophisticated multi-stage research architecture
that breaks down complex questions into sub-questions, researches them,
analyzes findings, and synthesizes comprehensive answers.
"""

from model import llm_basic, llm_advanced
from tavily import TavilyClient
import os
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_queries(plan: str, max_queries: int = 5) -> List[str]:
    """
    Extract individual research queries from a planning output.
    
    Args:
        plan: The planning output containing sub-questions
        max_queries: Maximum number of queries to extract
        
    Returns:
        List of extracted queries
    """
    if not plan:
        logger.warning("Empty plan provided to extract_queries")
        return []
    
    try:
        # Use LLM to extract structured queries from the plan
        extraction_prompt = f"""
        From the following research plan, extract EXACTLY {max_queries} distinct,
        clear, and specific research queries. Return ONLY the queries, one per line,
        without numbering or bullet points.
        
        Plan:
        {plan}
        """
        
        response = llm_basic.invoke(extraction_prompt).content
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        
        # Limit to max_queries
        queries = queries[:max_queries]
        
        logger.info(f"Extracted {len(queries)} queries from plan")
        return queries
    except Exception as e:
        logger.error(f"Error extracting queries: {e}")
        return []


class PlanModel(BaseModel):
    plan_text: str = Field(..., description="Full planning narrative to guide research")
    sub_questions: List[str] = Field(..., description="List of specific research queries (4-6 items)")


def web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Perform a web search for the given query using Tavily API.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary containing search results and metadata
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to web_search")
        return {"query": query, "results": [], "error": "Empty query"}
    
    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            #topic="news"  # Focus on current information
        )
        
        logger.info(f"Search completed for '{query}': {len(response.get('results', []))} results")
        return {
            "query": query,
            "results": response.get("results", []),
            "raw_response": response
        }
    except Exception as e:
        logger.error(f"Error during web search for '{query}': {e}")
        return {
            "query": query,
            "results": [],
            "error": str(e)
        }


def extract_sources(search_results: List[Dict[str, Any]]) -> List[str]:
    """Extract a deduplicated list of source citations from search results.

    Each citation is formatted as: 'Title — URL' when possible.
    """
    seen = set()
    sources = []
    for entry in search_results:
        results = entry.get("results", [])
        for r in results:
            # Try common fields for URL and title
            url = r.get("url")
            title = r.get("title")

            if url:
                key = url
            else:
                # fallback to title-based dedupe
                key = title

            if key in seen:
                continue
            seen.add(key)

            if url:
                sources.append(f"- {title} — {url}")
            else:
                sources.append(f"- {title}")

    return sources


def validate_state(state: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate that the state contains required keys.
    
    Args:
        state: The state dictionary to validate
        required_keys: List of required keys
        
    Returns:
        True if valid, False otherwise
    """
    for key in required_keys:
        if key not in state or state[key] is None:
            logger.warning(f"Missing required key in state: {key}")
            return False
    return True


# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================

def planner_fn(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planning Phase: Break down the question into research sub-questions.
    
    This function analyzes the input question and creates a structured
    research plan with specific sub-questions to investigate.
    
    Args:
        state: Current state containing 'question'
        
    Returns:
        Updated state with 'plan' and 'sub_questions'
    """
    if not validate_state(state, ["question"]):
        return {"plan": "", "sub_questions": []}
    
    question = state["question"].strip()
    logger.info(f"Planning research for question: {question[:100]}...")
    
    try:
        # Use structured output to get both the plan text and the sub-questions
        system_msg = (
            "You are a research expert. Create a comprehensive research plan and a list of 4-6 specific sub-questions "
            "that will guide a deep research process. The response must follow the PlanModel schema exactly."
        )

        user_msg = f"Question: {question}\n\nProvide a detailed plan and an explicit list of sub-questions."

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", user_msg)
        ])

        try:
            structured = llm_basic.with_structured_output(PlanModel, strict=True).invoke(prompt.format_messages())
            plan_text = structured.plan_text
            sub_questions = structured.sub_questions

            logger.info(f"Planning complete: {len(sub_questions)} sub-questions identified (structured)")
            return {"plan": plan_text, "sub_questions": sub_questions}
        except Exception as se:
            # Fallback: two-step approach (previous behavior)
            logger.warning(f"Structured planner failed: {se}; falling back to plain-text plan + extraction")
            planning_prompt = f"""
            You are a research expert. Analyze this question and create a comprehensive
            research plan with 4-6 specific sub-questions that will help answer it thoroughly.
            
            Focus on:
            1. Core concepts that need definition
            2. Key facts and statistics needed
            3. Different perspectives or viewpoints
            4. Recent developments or trends
            5. Practical implications
            
            Question: {question}
            
            Create a structured plan that will guide deep research.
            """

            plan = llm_basic.invoke(planning_prompt).content
            sub_questions = extract_queries(plan, max_queries=6)

            logger.info(f"Planning complete: {len(sub_questions)} sub-questions identified (fallback)")
            return {"plan": plan, "sub_questions": sub_questions}
    except Exception as e:
        logger.error(f"Error in planning phase: {e}")
        return {"plan": "", "sub_questions": [], "error": str(e)}


def search_fn(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search Phase: Execute web searches for all sub-questions.
    
    Performs targeted web searches for each sub-question to gather
    comprehensive information from multiple sources.
    
    Args:
        state: Current state containing 'sub_questions'
        
    Returns:
        Updated state with 'search_results'
    """
    if not validate_state(state, ["sub_questions"]):
        return {"search_results": []}
    
    sub_questions = state.get("sub_questions", [])
    
    if not sub_questions:
        logger.warning("No sub-questions provided for search phase")
        return {"search_results": []}
    
    logger.info(f"Starting search phase: {len(sub_questions)} queries to execute")
    
    search_results = []
    
    try:
        for i, query in enumerate(sub_questions, 1):
            logger.info(f"Searching [{i}/{len(sub_questions)}]: {query[:80]}...")
            result = web_search(query, max_results=3)
            search_results.append(result)
        
        logger.info(f"Search phase complete: {len(search_results)} results collected")
        
        return {"search_results": search_results}
    except Exception as e:
        logger.error(f"Error in search phase: {e}")
        return {
            "search_results": search_results,
            "error": str(e)
        }


def analyze_fn(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analysis Phase: Extract insights and synthesize findings.
    
    Analyzes search results to identify patterns, contradictions,
    key findings, and evidence for comprehensive understanding.
    
    Args:
        state: Current state containing 'search_results' and original 'question'
        
    Returns:
        Updated state with 'insights'
    """
    if not validate_state(state, ["search_results"]):
        return {"insights": ""}
    
    search_results = state.get("search_results", [])
    question = state.get("question", "")
    
    logger.info(f"Analyzing {len(search_results)} search results...")
    
    try:
        # Format search results for analysis
        formatted_results = []
        for result in search_results:
            query = result.get("query", "")
            results = result.get("results", [])
            formatted_results.append(f"\n=== Results for '{query}' ===\n")
            for r in results[:3]:  # Take top 3 results per query
                formatted_results.append(f"Title: {r.get('title', 'N/A')}\n")
                formatted_results.append(f"Content: {r.get('content', '')}\n")
        
        analysis_prompt = f"""
        Analyze the following research findings to answer this question:
        "{question}"
        
        Research Findings:
        {''.join(formatted_results)}
        
        Provide a detailed analysis that includes:
        1. Key findings and facts discovered
        2. Consensus vs. differing viewpoints
        3. Important statistics or data points
        4. Any gaps in information
        5. Reliability assessment of sources
        
        Focus on extracting actionable insights for a comprehensive answer.
        """
        
        insights = llm_advanced.invoke(analysis_prompt).content
        
        logger.info("Analysis complete: Insights extracted")
        
        return {"insights": insights}
    except Exception as e:
        logger.error(f"Error in analysis phase: {e}")
        return {
            "insights": "",
            "error": str(e)
        }


def synthesize_fn(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesis Phase: Create final structured answer.
    
    Synthesizes all research findings into a comprehensive, well-structured,
    and thoroughly reasoned answer to the original question.
    
    Args:
        state: Current state containing 'insights', 'question', and 'sub_questions'
        
    Returns:
        Updated state with 'answer'
    """
    if not validate_state(state, ["insights"]):
        return {"answer": ""}
    
    insights = state.get("insights", "")
    question = state.get("question", "")
    sub_questions = state.get("sub_questions", [])
    
    logger.info("Starting synthesis phase...")
    
    try:
        synthesis_prompt = f"""
        Based on the following research insights, provide a comprehensive and structured answer.
        
        Original Question: {question}
        
        Sub-questions Investigated:
        {chr(10).join(f'• {q}' for q in sub_questions)}
        
        Research Insights:
        {insights}
        
        Structure your answer as follows:
        1. Executive Summary (2-3 sentences)
        2. Detailed Explanation (with relevant evidence)
        3. Key Findings (bullet points)
        4. Different Perspectives (if applicable)
        5. Conclusion and Implications
        
        Ensure the answer is well-reasoned, evidence-based, and directly addresses the question.
        """
        
        answer = llm_advanced.invoke(synthesis_prompt).content
        
        # Attach sources extracted from search_results (if any) at the end of the answer
        sources = extract_sources(state.get("search_results", []))
        if sources:
            answer_with_sources = f"{answer}\n\nSources:\n{chr(10).join(sources)}"
        else:
            answer_with_sources = answer

        logger.info("Synthesis complete: Final answer generated (with sources appended)")

        return {"answer": answer_with_sources}
    except Exception as e:
        logger.error(f"Error in synthesis phase: {e}")
        return {
            "answer": "",
            "error": str(e)
        }


# ============================================================================
# OPTIONAL REFINEMENT PHASE
# ============================================================================

def refinement_fn(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional Refinement Phase: Improve answer quality with additional context.
    
    This phase can identify gaps in the answer and perform targeted
    searches to fill them, ensuring comprehensive coverage.
    
    Args:
        state: Current state containing 'answer'
        
    Returns:
        Updated state with 'refined_answer'
    """
    if not validate_state(state, ["answer"]):
        return {"refined_answer": state.get("answer", "")}
    
    answer = state.get("answer", "")
    question = state.get("question", "")
    
    logger.info("Starting refinement phase...")
    
    try:
        refinement_prompt = f"""
        Review this answer to the question and identify any gaps or areas that need more detail.
        
        Question: {question}
        
        Current Answer:
        {answer}
        
        Identify 2-3 specific areas where additional research or clarification would strengthen
        the answer. Be concise.
        """
        
        gaps = llm_basic.invoke(refinement_prompt).content
        
        # If significant gaps identified, perform targeted searches
        gap_queries = extract_queries(gaps, max_queries=3)
        
        if gap_queries:
            logger.info(f"Identified {len(gap_queries)} gaps to fill with additional search")
            gap_results = []
            for query in gap_queries:
                result = web_search(query, max_results=2)
                gap_results.append(result)
            
            # Enhance original answer with gap-filling results
            enhancement_prompt = f"""
            Enhance this answer with additional information from these targeted searches:
            
            Original Answer:
            {answer}
            
            Additional Research:
            {gap_results}
            
            Improve the answer by incorporating the most relevant findings.
            """
            
            refined_answer = llm_advanced.invoke(enhancement_prompt).content
            
            logger.info("Refinement complete: Answer enhanced with additional research")
            
            return {"refined_answer": refined_answer}
        else:
            logger.info("No significant gaps identified in answer")
            return {"refined_answer": answer}
    except Exception as e:
        logger.error(f"Error in refinement phase: {e}")
        return {
            "refined_answer": state.get("answer", ""),
            "error": str(e)
        }