from agent_factory import create_llm, invoke_text
from model import llm_advanced, llm_basic

def direct_architecture(state) -> str:
    llm = create_llm(llm_basic, state["tools"])

    improved_text = ""

    if "web_search" in state["tools"]:
        improved_text = "If you use web_search results, include a final section called References with the exact source URLs you used."

    answer = invoke_text(
        llm,
        f"""
        Answer this question directly and vaguely, without a clear structure or sections. Focus on providing a quick answer rather than a detailed explanation.:
        {state['question']}

        {improved_text}
        """,
    )

    return {"answer": answer}