from evaluation import evaluate
from direct_arch import direct_architecture
from mutation import init_genome
from graph import build_agent

genome = init_genome()

genome["architecture"] = "planner_executor"

#genome["tools"] = ["web_search", "summarize"]

agent = build_agent(genome)

question = "What are transformers in machine learning? Do a deep research and provide a comprehensive answer."

result = agent.invoke({"question": question, "tools": genome["tools"]})
answer = result["answer"]

score, feedback = evaluate(question, answer)

print(f"Score: {score}\nFeedback: {feedback}\nResponse: {answer}")