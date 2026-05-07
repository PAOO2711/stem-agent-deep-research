from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from model import llm_basic, llm_advanced
from tools import web_search, summarize

TOOL_REGISTRY = {
    "web_search": web_search,
    "summarize": summarize,
}


def tools_from_genome(tool_names):
    return [TOOL_REGISTRY[name] for name in tool_names]


def create_llm(llm, tool_names):
    selected_tools = tools_from_genome(tool_names)
    if selected_tools:
        return create_agent(llm, tools=selected_tools)
    return llm


def invoke_text(llm, prompt: str) -> str:
    if hasattr(llm, "steps"):
        result = llm.invoke({"messages": [HumanMessage(content=prompt)]})
        return result["messages"][-1].content

    response = llm.invoke(prompt)
    return response.content