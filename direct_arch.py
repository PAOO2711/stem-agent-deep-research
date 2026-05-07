from agent_factory import create_llm, invoke_text
from model import llm_advanced, llm_basic

def direct_architecture(state) -> str:
    llm = create_llm(llm_basic, state["tools"])

    answer = invoke_text(
        llm,
        f"""
        Answer this question directly:
        {state['question']}

        If you use web_search results, include a final section called References
        with the exact source URLs you used.
        """,
    )

    return {"answer": answer}