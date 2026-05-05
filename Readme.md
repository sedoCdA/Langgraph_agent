# LangGraph Multi-Agent System

A multi-agent AI pipeline built with LangGraph and LangChain. The system breaks down a high-level goal into tasks, executes each task with web search assistance, and iteratively verifies the quality of results until they meet a defined standard.

---

## Overview

This project demonstrates a Plan-Execute-Verify agentic loop using three specialized agents:

- **Planner** - Receives a user-defined goal and decomposes it into a list of at most five concrete, actionable tasks.
- **Executor** - Iterates over each task, performs a live DuckDuckGo web search for real-time context, and produces a detailed result using a large language model.
- **Verifier** - Evaluates all results against the original goal using a rubric covering completeness, accuracy, and clarity. If the combined score falls below the threshold, it sends the results back to the executor with a critique. The loop continues until results are approved or a maximum of three iterations is reached.

The workflow is orchestrated as a directed state graph using LangGraph, with conditional edges to handle the approve-or-retry logic automatically.

---

## Architecture

```
START --> Planner --> Executor --> Verifier
                         ^            |
                         |            | (if not approved)
                         +------------+
                                      |
                                      | (if approved or max iterations)
                                     END
```

### Agent Roles

| Agent    | Role                                                                 |
|----------|----------------------------------------------------------------------|
| Planner  | Decomposes the goal into a JSON list of tasks                        |
| Executor | Completes each task using LLM reasoning and live web search results  |
| Verifier | Scores results and decides to approve or request improvement         |

---

## Tech Stack

| Library                  | Purpose                                      |
|--------------------------|----------------------------------------------|
| LangChain                | Core abstractions for messages and LLM calls |
| LangGraph                | State machine and graph orchestration        |
| langchain-groq           | Groq-hosted LLM inference (Llama 3.1 8B)    |
| langchain-community      | DuckDuckGo search tool integration           |
| python-dotenv            | Loading API keys from the .env file          |

---

## Project Structure

```
Langchain agent/
    agent1.py       - Main script containing all agents and the graph definition
    .env            - Environment variables (API keys, not committed to git)
    .gitignore      - Excludes .env and .venv from version control
    Readme.md       - This file
```

---

## Prerequisites

- Python 3.10 or higher
- A Groq API key (free tier available at https://console.groq.com)

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "Langchain agent"
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install langchain langgraph langchain-groq langchain-community python-dotenv duckduckgo-search
```

### 4. Configure your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Do not commit this file. It is already listed in `.gitignore`.

---

## Running the Agent

```bash
python agent1.py
```

The agent will run with the default goal defined in `initial_state` at the bottom of `agent1.py`:

```python
"goal": "Research and summarize the top 3 trends in generative ai for 2025"
```

To change the goal, edit the `goal` field in `initial_state` before running.

---

## Sample Output

```
[Planner] Generated 4 tasks:
  1. Research top generative AI trends in 2025
  2. Summarize multimodal AI advancements
  ...

[Executor] Task: Research top generative AI trends in 2025
  Generative AI in 2025 is defined by three major shifts...

[Verifier] Score: 0.87, Approved: True
  Results are comprehensive and well-structured.

Completed in 1 iteration(s)
```

---

## Verifier Scoring Rubric

The Verifier agent scores each execution round using the following criteria:

| Criterion    | Weight | Description                                          |
|--------------|--------|------------------------------------------------------|
| Completeness | 0 - 0.4 | Does the output fully address the original goal?    |
| Accuracy     | 0 - 0.3 | Is the information correct and trustworthy?         |
| Clarity      | 0 - 0.3 | Is the response well-structured and readable?       |

A combined score of 1.0 is the maximum. Results are approved when the LLM verifier judges them sufficient. If not approved, the critique is passed back to the executor for the next iteration. The loop exits after three iterations regardless of approval status.

---

## Customization

- **Change the LLM model** - Edit the `model` parameter in the `ChatGroq` constructor in `agent1.py`.
- **Change the goal** - Update the `goal` key in `initial_state`.
- **Adjust max iterations** - Change the threshold value in the verifier function (`>= 3`).
- **Limit task count** - The planner prompt instructs the LLM to generate at most five tasks. Edit the system prompt in `planner()` to change this.

---

## Security Note

Your Groq API key is a secret credential. Never commit the `.env` file to a public repository. The `.gitignore` in this project already excludes it.

---

## License

This project is open source and available under the MIT License.
