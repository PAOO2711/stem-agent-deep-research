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
    # Agents created with create_agent expect a dict state with messages.
    try:
        result = llm.invoke({"messages": [HumanMessage(content=prompt)]})
        if isinstance(result, dict) and "messages" in result:
            return result["messages"][-1].content
    except Exception:
        pass

    # Plain chat models accept string/message input and return AIMessage.
    response = llm.invoke(prompt)
    return response.content