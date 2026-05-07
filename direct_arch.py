from agent_factory import create_llm, invoke_text
from model import llm_advanced

def direct_architecture(state) -> str:
    llm = create_llm(llm_advanced, state["tools"])

    answer = invoke_text(llm, f"Answer this question directly: {state['question']}")

    return answer