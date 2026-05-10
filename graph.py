from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional, Dict, Any
from direct_arch import direct_architecture
from planner_executor_arch import (
    planner_fn, search_fn, analyze_fn, synthesize_fn, refinement_fn
)
from reflection import reflect_fn

class AgentState(TypedDict, total=False):
    """
    Complete state schema for the research agent.
    
    Fields:
        question: The original research question
        tools: List of available tools
        plan: Research plan breakdown
        sub_questions: Extracted sub-questions for research
        search_results: Results from web searches
        documents: Legacy field for backwards compatibility
        insights: Analyzed findings and patterns
        answer: Final synthesized answer
        refined_answer: Enhanced answer after refinement
        error: Error message if any phase fails
    """
    question: str
    tools: List[str]
    plan: str
    sub_questions: Optional[List[str]]
    search_results: Optional[List[Dict[str, Any]]]
    documents: Optional[List[str]]  # Legacy field
    insights: Optional[str]
    answer: Optional[str]
    refined_answer: Optional[str]
    error: Optional[str]


def build_agent(genome):
    """
    Build agent graph based on genome configuration.
    
    Args:
        genome: Configuration dictionary with:
            - architecture: "direct" or "planner_executor"
            - reflection: Whether to include reflection phase
            - refinement: Whether to include refinement phase (for planner_executor)
    
    Returns:
        Compiled graph ready for execution
    """
    graph = StateGraph(AgentState)

    if genome.architecture == "direct":
        # Direct architecture: single-pass response
        graph.add_node("answer", direct_architecture)
        graph.add_edge(START, "answer")
        graph.add_edge("answer", END)

    else:
        # Planner-Executor architecture: multi-phase research pipeline
        graph.add_node("planner", planner_fn)
        graph.add_node("search", search_fn)
        graph.add_node("analyze", analyze_fn)
        graph.add_node("synthesize", synthesize_fn)

        # Core pipeline flow
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "search")
        graph.add_edge("search", "analyze")
        graph.add_edge("analyze", "synthesize")

        # Optional refinement phase
        if genome.refinement:
            graph.add_node("refinement", refinement_fn)
            graph.add_edge("synthesize", "refinement")
            final_node = "refinement"
        else:
            final_node = "synthesize"

        # Optional reflection phase
        if genome.reflection:
            graph.add_node("reflect", reflect_fn)
            graph.add_edge(final_node, "reflect")
            graph.add_edge("reflect", END)
        else:
            graph.add_edge(final_node, END)

    return graph.compile()
