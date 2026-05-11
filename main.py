from graph import build_agent
from mutation import init_genome, mutate
from evaluation import evaluate, stop_condition


def extract_final_answer(result):
    """Extract the final answer text from graph output state."""
    if isinstance(result, dict):
        return result.get("refined_answer") or result.get("answer") or ""
    return str(result)

def main():
    # Initialize the agent's genome
    genome = init_genome()

    # Example question to test the agent
    question = input("Enter a question for deep research: ")

    question += ".Do a deep research and provide a comprehensive answer."

    score = 0

    best_score = 0
    best_answer = ""
    best_genome = genome
    
    for i in range(5):  
        # Build the agent based on the genome
        agent = build_agent(genome)

        # Graph returns full state; extract only the final answer text
        result = agent.invoke({"question": question, "tools": genome.tools})
        answer = extract_final_answer(result)

        score, evaluation_score, feedback = evaluate(question, answer)

        
        
        if hasattr(feedback, "model_dump"):
            feedback_display = feedback.model_dump()
        elif isinstance(feedback, dict):
            feedback_display = feedback
        else:
            feedback_display = str(feedback)

        print(f"Score: {score}\nEvaluation: {evaluation_score}\nFeedback: {feedback_display}\n")

        # Keep track of the best answer and genome configuration
        if score >= best_score:
            best_score = score
            best_answer = answer
            best_genome = genome
        else:
            print("-------ROLLBACK TO BEST GENOME-------\n")

        if stop_condition(best_score):
            break
    
        # Update genome based on feedback
        print("---------MUTATING GENOME BASED ON FEEDBACK---------\n")
        new_genome = mutate(best_genome, feedback)
        print(f"New Genome Configuration:\n{new_genome}\n")
        print("--------------------------------------------------\n")
        genome = new_genome
    
    
    print(f"Question: {question}")
    print(f"Answer: {best_answer}")

if __name__ == "__main__":
    main()