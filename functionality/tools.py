from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
import os
from models.model import llm_basic, llm_advanced

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Perform a web search for the given query."""

    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
    )

    results = response.get("results", [])
    if not results:
        return "No se encontraron resultados para esta busqueda."

    snippets = []
    references = []

    for i, item in enumerate(results, start=1):
        title = item.get("title", "Sin titulo")
        url = item.get("url", "")
        content = (item.get("content", "") or "").strip().replace("\n", " ")

        snippets.append(f"[{i}] {title}: {content}")
        if url:
            references.append(f"[{i}] {url}")

    return (
        "Resultados de busqueda:\n"
        + "\n".join(snippets)
        + "\n\nReferencias:\n"
        + "\n".join(references)
    )

@tool
def summarize(text: str) -> str:
    """Summarize the given text."""

    summary = llm_advanced.invoke(f"Summarize this text: {text}")

    return summary.content
