from model import LLM

def planner_fn(state):
    question = state["question"]

    plan = llm(f"""
    Break this question into sub-questions for deep research:
    {question}
    """)

    return {"plan": plan}

def search_fn(state):
    queries = extract_queries(state["plan"])

    results = []
    for q in queries:
        results.append(web_search(q))

    return {"documents": results}

def analyze_fn(state):
    docs = state["documents"]

    insights = llm(f"""
    Extract key insights and compare sources:
    {docs}
    """)

    return {"insights": insights}

def synthesize_fn(state):
    insights = state["insights"]

    answer = llm(f"""
    Write a structured, well-reasoned answer:
    {insights}
    """)

    return {"answer": answer}