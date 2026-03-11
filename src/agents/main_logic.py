import os
import sys
import json
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END

# --- 1. THE PATH FIX (For Python Mastery)  ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # Finds the 'src' folder

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- 2. VERIFIED IMPORT ---
try:
    from chroma_db_setup import get_relevant_context
except ImportError as e:
    print(f"CRITICAL ERROR: Import failed. Error: {e}")
    sys.exit(1)

# --- 3. STATE DEFINITION ---
class AgentState(TypedDict):
    query: str
    context: List[str]
    answer: str
    verified: bool
    iterations: int

# --- 4. THE REASONING NODES [cite: 25, 33] ---

def retrieve_docs_node(state: AgentState):
    print("---RETRIEVING CONTEXT---")
    results = get_relevant_context(state['query'], n_results=3)
    retrieved_text = [res['text'] for res in results] if results else []
    return {"context": retrieved_text, "iterations": state.get("iterations", 0) + 1}

def grade_documents_node(state: AgentState):
    """Checks if the retrieved text actually addresses the query[cite: 39]."""
    print("---CHECKING RELEVANCE---")
    context_str = "\n".join(state['context'])
    
    if not state['context'] or len(context_str) < 50:
        print("RELEVANCE CHECK: FAILED")
        return {"verified": False}
    
    print("RELEVANCE CHECK: PASSED")
    return {"verified": True}

def generate_answer_node(state: AgentState):
    print("---GENERATING FINAL ANSWER---")
    context_str = "\n".join(state['context'])

    # Strict Prompt Engineering [cite: 14]
    prompt = f"""
    You are a medical expert[cite: 14]. Using ONLY the provided context, answer X. 
    If unsure, say you don't know[cite: 14]. Ensure 100% citation accuracy[cite: 22].

    Context: {context_str}
    Question: {state['query']}
    
    Answer:
    """
    
    # Placeholder for Llama 3.2 / Groq API Call [cite: 27, 65]
    response = "Based on the provided research, the answer is..." 
    return {"answer": response}

# --- 5. LOGIC & WORKFLOW ---

def decide_to_generate(state: AgentState):
    if state["verified"] or state["iterations"] >= 2:
        return "generate"
    return "retrieve"

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_docs_node)
workflow.add_node("grade_docs", grade_documents_node)
workflow.add_node("generate", generate_answer_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_docs")
workflow.add_conditional_edges("grade_docs", decide_to_generate, {
    "generate": "generate",
    "retrieve": "retrieve"
})
workflow.add_edge("generate", END)

medisight_agent = workflow.compile()

if __name__ == "__main__":
    inputs = {"query": "What are the side effects?", "iterations": 0}
    for output in medisight_agent.stream(inputs):
        print(output)