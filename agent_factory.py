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


def _has_sources_section(text: str) -> bool:
    return re.search(r"(?im)^\s*(?:#{1,6}\s*)?sources\s*:?\s*$", text) is not None


def _strip_trailing_references_section(text: str) -> str:
    return re.sub(
        r"\n{2,}(?:#{1,6}\s*)?references\s*:?\s*\n.*\Z",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def invoke_text(llm, prompt: str) -> str:
    # Agents created with create_agent expect a dict state with messages.
    try:
        result = llm.invoke({"messages": [HumanMessage(content=prompt)]})
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            answer = messages[-1].content

            # Print tool calls and tool results for debugging/visibility
            for msg in messages:
                # model decided to call a tool
                if getattr(msg, "tool_calls", None):
                    try:
                        print("Tool call:", msg.tool_calls)
                    except Exception:
                        print("Tool call: [could not display tool_calls]")

                # tool executed and returned a ToolMessage
                """ if type(msg).__name__ == "ToolMessage":
                    name = getattr(msg, "name", None)
                    content = str(getattr(msg, "content", ""))
                    print(f"Tool executed: {name}")
                    print("Output (first 300 chars):", content[:300]) """

            references = []
            for message in messages:
                if type(message).__name__ == "ToolMessage":
                    content = str(getattr(message, "content", ""))
                    references.extend(_extract_urls(content))

            unique_references = list(dict.fromkeys(references))
            answer = _strip_trailing_references_section(answer)

            if unique_references and not _has_sources_section(answer):
                answer += "\n\nSources:\n" + "\n".join(unique_references)

            return answer
    except Exception:
        pass

    # Plain chat models accept string/message input and return AIMessage.
    response = llm.invoke(prompt)
    return response.content