from model import llm_basic, llm_advanced

def reflect_fn(state):
    answer = state["answer"]

    improved = llm_advanced.invoke(f"""
    You are reviewing an answer.

    Improve it by:
    - fixing mistakes
    - adding missing details
    - improving structure

    Original answer:
    {answer}
    """)

    return {"answer": improved}