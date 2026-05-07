from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
import re

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


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s)\]>,]+", text)


def invoke_text(llm, prompt: str) -> str:
    # Agents created with create_agent expect a dict state with messages.
    try:
        result = llm.invoke({"messages": [HumanMessage(content=prompt)]})
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            answer = messages[-1].content

            references = []
            for message in messages:
                if type(message).__name__ == "ToolMessage":
                    content = str(getattr(message, "content", ""))
                    references.extend(_extract_urls(content))

            unique_references = list(dict.fromkeys(references))
            if unique_references and "references:" not in answer.lower():
                answer += "\n\nReferences:\n" + "\n".join(unique_references)

            return answer
    except Exception:
        pass

    # Plain chat models accept string/message input and return AIMessage.
    response = llm.invoke(prompt)
    return response.content