from model import llm_advanced

def direct_architecture(query: str) -> str:
    answer = llm_advanced.invoke(f"Answer this question directly: {query}")
    
    return answer

