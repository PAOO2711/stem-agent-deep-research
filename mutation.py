from pydantic import BaseModel, Field
from typing import Literal, List
from model import llm_advanced

class Genome(BaseModel):
    """
    Agent genome configuration.
    
    Attributes:
        architecture: The core architecture of the agent
            - "direct": Simple single-pass response
            - "planner_executor": Multi-phase research pipeline
        tools: List of available tools for the agent
        reflection: Whether to use reflection phase for self-improvement
        refinement: Whether to use refinement phase (only for planner_executor)
    """
    architecture: Literal["direct", "planner_executor"] = Field(
        default="direct",
        description="The architecture of the agent"
    )
    tools: List[Literal["web_search", "summarize"]] = Field(
        default=[],
        description="List of tools the agent can use"
    )
    reflection: bool = Field(
        default=False,
        description="Whether the agent uses reflection for self-improvement"
    )
    refinement: bool = Field(
        default=False,
        description="Whether the agent uses refinement phase (planner_executor only)"
    )

def init_genome():
    """Initialize a default genome configuration."""
    return {
        "architecture": "direct",
        "tools": [],
        "reflection": False,
        "refinement": False,
    }

def mutate(genome, feedback):
    """
    Generate an improved genome based on feedback from evaluation.
    
    Args:
        genome: Current genome configuration
        feedback: Evaluation feedback to guide mutation
        
    Returns:
        New mutated genome
    """

    prompt = f"""
    Current agent configuration:
    {genome}

    Feedback:
    {feedback}

    Propose an improved agent configuration based on the feedback.

    Options for architecture:
    - 'direct': A simple architecture that answers questions directly.
    - 'planner_executor': An architecture that plans and executes multiple research steps 
      to answer the question thoroughly (best for complex questions requiring deep research).

    You can also choose to add tools to the agent. Available tools include:
    - 'web_search': A tool for searching the web.
    - 'summarize': A tool for summarizing text.

    Enhancement options:
    - 'reflection': Whether the agent should use reflection for self-improvement after 
      answering questions. Recommended for learning from mistakes.
    - 'refinement': Whether to add a refinement phase to identify and fill gaps in the 
      research (only applicable with planner_executor architecture). Recommended for 
      questions requiring comprehensive coverage.

    For deep research questions, consider using planner_executor architecture with 
    refinement=True for best results.
    """
    
    new_genome = llm_advanced.with_structured_output(
        Genome,
        strict=True,
    ).invoke(prompt)
    
    
    return new_genome