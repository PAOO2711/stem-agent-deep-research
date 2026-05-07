from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional
from direct_arch import direct_architecture
from planner_executor_arch import planner_fn, search_fn, analyze_fn, synthesize_fn
from reflection import reflect_fn

class AgentState(TypedDict, total=False):
    question: str
    tools: List[str]
    plan: Optional[str]
    documents: Optional[List[str]]
    insights: Optional[str]
    answer: Optional[str]


def build_agent(genome):
    graph = StateGraph(AgentState)


    if genome["architecture"] == "direct":
        graph.add_node("answer", direct_architecture)

        graph.add_edge(START, "answer")
        graph.add_edge("answer", END)

    else:
        graph.add_node("planner", planner_fn)
        graph.add_node("search", search_fn)
        graph.add_node("analyze", analyze_fn)
        graph.add_node("synthesize", synthesize_fn)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "search")
        graph.add_edge("search", "analyze")
        graph.add_edge("analyze", "synthesize")

        if genome["reflection"]:
            graph.add_node("reflect", reflect_fn)
            graph.add_edge("synthesize", "reflect")
            graph.add_edge("reflect", END)
        else:
            graph.add_edge("synthesize", END)
    

    return graph.compile()

