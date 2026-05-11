# Stem agent for deep research

## Setup instructions

Create a `.env` file at the project root with the same keys as `.env.example`:

```env
TAVILY_API_KEY=your-tavily-api-key
OPENAI_API_KEY=your-openai-api-key
```

Use a virtual environment and install the dependencies from `requirements.txt`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the project with:

```bash
python3 main.py
```

## Evaluation

The final response is evaluated on a 0-10 scale using these components:

```text
correctness  = 0-2 points
depth        = 0-2 points
structure    = 0-2 points
sources      = 0-2 points
bonus        = 0-2 points
penalties    = deducted from the total

final_score = correctness + depth + structure + sources + bonus - abs(penalties)
final_score is clamped between 0 and 10
```

The evaluator also returns structured feedback signals used by the mutation step, such as:

```text
missing_information
needs_multi_step_reasoning
poor_structure
research_gaps
weak_sources
hallucinations
severity
feedback_text
```

The loop stops early when the score is greater than 8.

## Experiments

The repository includes 4 experiments based on open-ended deep research questions.

Each experiment contains the execution output of the evolutionary process, including:
- agent configurations (genomes),
- evaluation scores,
- feedback signals,
- architecture mutations,
- and the final evolved response.

