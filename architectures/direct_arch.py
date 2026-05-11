from models.agent_factory import create_llm, invoke_text
from models.model import llm_advanced, llm_basic

def direct_architecture(state) -> str:
    llm = create_llm(llm_basic, state["tools"])

    improved_text = ""

    if "web_search" in state["tools"]:
        improved_text = "If you use web_search results, include a final section called Sources with the exact source URLs you used."

    answer = invoke_text(
        llm,
        f"""
        Answer this question directly:
        {state['question']}

        {improved_text}
        """,
    )

    return {"answer": answer}