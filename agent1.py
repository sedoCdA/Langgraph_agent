import os, json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, List
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")
search = DuckDuckGoSearchRun()


class AgentState(TypedDict):
    goal: str
    tasks: List[str]
    results: List[str]
    critique: str
    approved: bool
    iterations: int

#-------------------- Planner Agent-------------------------------

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

    return {**state, "tasks": tasks}

#--------------------------Executor Agent-------------------------------

def executor(state: AgentState):
    results = []
    critique_ctx = ""
    if state["critique"]:
        critique_ctx = f"Your previous attempt was rejected. Critique: {state['critique']}\nImprove your output accordingly."

    for task in state["tasks"]:
        system = f"You are an execution agent. Complete the task below thoroughly. Use web search if you need current information.{critique_ctx}"

        search_ctx = ""
        try:
            search_result = search.run(task[:100])
            search_ctx = f"Web result:\n{search_result[:8000]}\n"
        except:
            pass

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"{search_ctx}\nTask: {task}")
        ]
        result = llm.invoke(messages).content
        results.append(result)
        print(f"\n[Executor] Task: {task[:60]}\n{result[:120]}")

    return {**state, "results": results, "iterations": state["iterations"] + 1}


#--------------------------Verifier Agent-------------------------------

def verifier(state:AgentState)-> AgentState:
    if state["iterations"] >= 3:
        print(f"[Verifier] Max iterations reached. Approving.")
        return {**state, "approved": True}
        
    combined_results = "\n\n".join(f"Task {i+1}: {t}\nResult:{r}"
        for i, (t, r) in enumerate(zip(state["tasks"],state["results"]))
        )
    system = """
    You are a quality verifier agent. Evaluate the results againts the original goal using this rubric:
    -Completeness: Does it fully address the goal? (0-0.4)
    -Accuracy: Is the information correct and trustworthy? (0-0.3)
    -Clarity: Is the response well-structured and easy to understand? (0-0.3)
    
    Sum the scores for a total between 0 and 1. 
    Respond only as a JSON:
    {"score": 0.85, "approved": True/False, "critique": "..."}"""
    
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Original Goal: {state['goal']}\n\nResults:\n{combined_results}")
    ]
    raw = llm.invoke(messages).content.strip()
    try:
        clean = raw.replace('```json', '').replace('```', '').strip()
        verdict = json.loads(clean)
        approved = verdict.get("approved", False)
        critique = verdict.get("critique", "")
        score = verdict.get("score", 0)
    except:
        approved, critique, score = False, raw, 0
    
    print(f"\n[Verifier] Score: {score:.2f}, Approved: {approved}\n{critique}")
    if not approved: print(f"Critique {critique}")
    return {**state, "approved": approved, "critique": critique}
    
#--------------------Build graph-----------------------------
graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("verifier", verifier)

graph.add_conditional_edges(
    "verifier",
    lambda state: END if state["approved"] else "executor",
)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "verifier")

app = graph.compile()

#--------------------Run It-----------------------------
initial_state: AgentState = {
    "goal": "Research and summarize the top 3 trends in generative ai for 2025",
    "tasks": [],
    "results": [],
    "critique": "",
    "approved": False,
    "iterations": 0
}

final_state = app.invoke(initial_state)

for i, (task, result) in enumerate(zip(final_state["tasks"], final_state["results"])):
    print(f"\n[Task {i+1}] {task}\n{result}")

print(f"\nCompleted in {final_state['iterations']} iteration(s)")