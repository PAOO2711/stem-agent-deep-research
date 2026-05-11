from functionality.evaluation import evaluate
from architectures.direct_arch import direct_architecture
from functionality.mutation import init_genome
from functionality.graph import build_agent

genome = init_genome()

#genome["tools"] = ["web_search", "summarize"]

agent = build_agent(genome)

question = "What are transformers in machine learning? Do a deep research and provide a comprehensive answer."

result = agent.invoke({"question": question, "tools": genome.tools})
answer = result["answer"]

score, evaluation_score, feedback = evaluate(question, answer)

print(f"Score: {score}\nEvaluation: {evaluation_score}\nFeedback: {feedback}\nResponse: {answer}")