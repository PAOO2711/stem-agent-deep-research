from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
import os
from model import llm_basic, llm_advanced

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Perform a web search for the given query."""
    
    response = tavily_client.search(
        query=query,
        search_depth="advanced",
    )
    
    return response

@tool
def summarize(text: str) -> str:
    """Summarize the given text."""

    summary = llm_advanced.invoke(f"Summarize this text: {text}")

    return summary.content
