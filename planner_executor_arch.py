from model import llm_basic, llm_advanced
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str) -> str:
    """Perform a web search for the given query."""
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=2
        )
    return response

def planner_fn(state):
    question = state["question"]

    plan = llm_basic.invoke(f"""
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

    insights = llm_advanced.invoke(f"""
    Extract key insights and compare sources:
    {docs}
    """)

    return {"insights": insights}

def synthesize_fn(state):
    insights = state["insights"]

    answer = llm_basic.invoke(f"""
    Write a structured, well-reasoned answer:
    {insights}
    """)

    return {"answer": answer}