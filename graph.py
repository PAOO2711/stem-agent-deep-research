from langgraph.graph import StateGraph, START, END

def build_agent(genome):
    graph = StateGraph(dict)

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
            graph.add_node("feedback", feedback)
            graph.add_edge("synthesize", "feedback")
            graph.add_edge("feedback", END)
        else:
            graph.add_edge("synthesize", END)
    

    return graph.compile()

