from pydantic import BaseModel, Field
from typing import Literal, List
from model import llm_advanced

class Genome(BaseModel):
    architecture: Literal["direct", "planner_executor"] = Field(default="direct", description="The architecture of the agent")
    tools: List[Literal["web_search", "summarize"]] = Field(default=[], description="List of tools the agent can use")
    reflection: bool = Field(default=False, description="Whether the agent uses reflection for self-improvement")

def init_genome():
    return {
        "architecture": "direct",
        "tools": [],
        "reflection": False,
    }

def mutate(genome, feedback):

    prompt = f"""
    Current agent configuration:
    {genome}

    Feedback:
    {feedback}

    Propose an improved agent configuration based on the feedback.

    Options for architecture:
    - 'direct': A simple architecture that answers questions directly.
    - 'planner_executor': An architecture that plans a sequence of steps to answer the question.

    You can also choose to add tools to the agent. Available tools include:
    - 'web_search': A tool for searching the web.
    - 'summarize': A tool for summarizing text.

    Reflection parameter indicates whether the agent should use reflection for self-improvement after answering questions.
    """
    
    new_genome = llm_advanced.with_structured_output(
        Genome,
        strict=True,
    ).invoke(prompt)
    
    
    return new_genome