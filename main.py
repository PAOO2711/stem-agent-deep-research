from graph import build_agent, Agent
from mutation import init_genome, mutate
from evaluation import evaluate, stop_condition

def main():
    # Initialize the agent's genome
    genome = init_genome()

    # Example question to test the agent
    question = input("Enter a question for deep research: ")
    
    for i in range(5):  
        # Build the agent based on the genome
        agent = build_agent(genome)

        # Get the agent's answer
        answer = agent.invoke({"question": question, "tools": genome["tools"]})

        score, feedback = evaluate(question, answer)

        if stop_condition(score):
            break
    
        # Update genome based on feedback
        new_genome = mutate(genome, feedback)
        genome = new_genome
    
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")