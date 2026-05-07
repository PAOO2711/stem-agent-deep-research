from evaluation import evaluate
from direct_arch import direct_architecture
from mutation import init_genome
from graph import build_agent

genome = init_genome()

genome["tools"] = ["web_search", "summarize"]

agent = build_agent(genome)

answer = agent.invoke({"question": question, "tools": genome["tools"]})

score, feedback = evaluate("What are transformers in machine learning?", answer)

print(f"Score: {score}\nFeedback: {feedback}\nResponse: {answer}")