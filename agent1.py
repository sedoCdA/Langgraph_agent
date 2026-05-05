import os, json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, List
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# Tools

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")
load_dotenv()


class AgentState(TypedDict):
    goal: str
    tasks: List[str]
    results: List[str]
    critique: str
    approved: bool
    iterations: int

search = DuckDuckGoSearchRun()

def planner(state: AgentState):
    system = """
    You are a planning agent. Break down the user's goal into smaller at most 5 concrete, actionable tasks. Respond only with a valid JSON array of strings. No preamble, no markdown.
    """
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Goal: {state['goal']}")
    ]
    response = llm.invoke(messages).content.strip()

    try:
        clean = response.replace('```json', '').replace('```', '').strip()
        tasks = json.loads(clean)
    except json.JSONDecodeError:
        tasks = [response]

    print(f"\n[Planner] Generated {len(tasks)} tasks:")
    for i, t in enumerate(tasks):
        print(f"  {i+1}. {t}")

    return{**state, "tasks": tasks}

graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_edge(START, "planner")
graph.add_edge("planner", END)

app = graph.compile()

initial_state : AgentState = {
    "goal": "Research and summarize the top 3 trends in generative ai for 2026",
    "tasks": [],
    "results": [],
    "critique": "",
    "approved": False,
    "iterations": 0
}
final_state = app.invoke(initial_state)

for i,(task, result) in enumerate(zip(final_state["tasks"], final_state["results"])):
    print(f"\n[Task {i+1}] {task}\n{result}")

print(f"\n Completed in {final_state['iterations']} iteration(s)")