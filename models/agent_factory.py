from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
import re

from models.model import llm_basic, llm_advanced
from functionality.tools import web_search, summarize

TOOL_REGISTRY = {
    "web_search": web_search,
    "summarize": summarize,
}


def tools_from_genome(tool_names):
    """Return a list of tool callables corresponding to the provided names.

    Args:
        tool_names: List of tool keys (e.g. 'web_search').

    Returns:
        A list of callables implementing the requested tools.
    """
    return [TOOL_REGISTRY[name] for name in tool_names]


def create_llm(llm, tool_names):
    """Create and return an LLM agent with the selected tools or return the LLM.

    If tool names are provided, constructs an agent using `create_agent`
    with the selected tools. Otherwise returns the `llm` instance unchanged.

    Args:
        llm: Base model instance (chat model or similar).
        tool_names: List of tool names to enable.

    Returns:
        Agent with tools configured or the original `llm` instance.
    """
    selected_tools = tools_from_genome(tool_names)
    if selected_tools:
        return create_agent(llm, tools=selected_tools)
    return llm


def _extract_urls(text: str) -> list[str]:
    """Extract all HTTP/HTTPS URLs found in `text`.

    Returns a list of URL strings found in the input text.
    """
    return re.findall(r"https?://[^\s)\]>,]+", text)


def _has_sources_section(text: str) -> bool:
    """Check whether the text already contains a 'Sources' section.

    This is used to avoid appending a duplicate sources section.
    """
    return re.search(r"(?im)^\s*(?:#{1,6}\s*)?sources\s*:?\s*$", text) is not None


def _strip_trailing_references_section(text: str) -> str:
    """Remove a trailing 'References' section from the text, if present.

    Returns the text with any final references block stripped.
    """
    return re.sub(
        r"\n{2,}(?:#{1,6}\s*)?references\s*:?\s*\n.*\Z",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def invoke_text(llm, prompt: str) -> str:
    """Invoke the LLM/agent with `prompt` and post-process the response.

    - If `llm` is an agent (returns a dict with messages), extract the final
      response, collect tool outputs, and append a 'Sources' section when needed.
    - If agent invocation fails, call the `llm` as a plain chat model.

    Args:
        llm: Model or agent instance.
        prompt: Prompt text to send.

    Returns:
        Final response text (may include a 'Sources' section).
    """
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