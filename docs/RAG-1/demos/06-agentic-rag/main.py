"""
Demo 6: Agentic RAG — ReAct Agent with Think-Act-Observe Loop
===============================================================
Implements a ReActAgent that reasons, retrieves documents, calculates,
and simulates web searches through a think-act-observe loop.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., str]


def retrieve_documents(query: str) -> str:
    """Simulated document retrieval."""
    corpus = {
        "rag": "RAG (Retrieval-Augmented Generation) combines retrieval with LLM generation.",
        "tfidf": "TF-IDF is a statistical measure of term importance in a document.",
        "bm25": "BM25 is a probabilistic ranking function for information retrieval.",
        "embedding": "Embeddings are dense vector representations of text.",
        "hybrid search": "Hybrid search combines sparse and dense retrieval for better results.",
        "transformer": "Transformers use self-attention mechanisms for sequence processing.",
    }
    query_lower = query.lower()
    results = []
    for keyword, content in corpus.items():
        if keyword in query_lower or query_lower in keyword:
            results.append(content)
    if not results:
        # Fuzzy fallback
        for keyword, content in corpus.items():
            if any(word in keyword for word in query_lower.split()):
                results.append(content)
    if not results:
        return f"No documents found for: {query}"
    return "\n".join(f"- {r}" for r in results)


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Only allow basic math
        safe = expression.replace(" ", "")
        if not re.match(r"^[\d\+\-\*\/\.\(\)]+$", safe):
            return "Error: expression contains invalid characters"
        result = eval(safe, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def search_web(query: str) -> str:
    """Simulated web search."""
    knowledge = {
        "population of china": "China population: approximately 1.41 billion (2023).",
        "capital of france": "The capital of France is Paris.",
        "python version": "Python 3.13 is the latest stable version.",
    }
    query_lower = query.lower().strip()
    for key, answer in knowledge.items():
        if key in query_lower or query_lower in key:
            return answer
    return f"Web search result for '{query}': (simulated) no results found."


TOOLS: List[Tool] = [
    Tool("retrieve_documents", "Retrieve documents from the knowledge base by query. Input: a search query string.", retrieve_documents),
    Tool("calculate", "Evaluate a mathematical expression. Input: a math expression like '2 + 2 * 3'.", calculate),
    Tool("search_web", "Search the web for current information. Input: a search query string.", search_web),
]


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

@dataclass
class ReActStep:
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None


class ReActAgent:
    """Agent that uses think-act-observe loop to answer questions."""

    def __init__(self, tools: List[Tool], max_steps: int = 5):
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.history: List[ReActStep] = []

    # ------------------------------------------------------------------
    # Sub-question decomposition
    # ------------------------------------------------------------------

    def _decompose_question(self, question: str) -> List[Dict]:
        """Split a compound question into sub-tasks."""
        tasks = []
        q = question.lower().strip().rstrip("?")

        # Check for "what is X and calculate Y" pattern
        parts = re.split(r"\band\b", q, maxsplit=1)
        if len(parts) == 2:
            left, right = parts
            # Detect if the right part has a calculation
            calc_match = re.search(r"calculate\s+(\d+\s*[\+\-\*\/]\s*\d+)", right)
            if calc_match:
                tasks.append({"type": "retrieve", "target": left.strip()})
                tasks.append({"type": "calculate", "target": calc_match.group(1).strip()})
                return tasks

        # Check for "calculate" or "compute" anywhere
        calc_match = re.search(r"(?:calculate|compute|what\s+is)\s+(\d+\s*[\+\-\*\/]\s*\d+)", q)
        if calc_match:
            tasks.append({"type": "calculate", "target": calc_match.group(1).strip()})
            return tasks

        # Default: single retrieval task
        tasks.append({"type": "retrieve", "target": question})
        return tasks

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _select_action(self, step: int, question: str, last_observation: str) -> Tuple[str, str]:
        """Decide which tool to call based on pending sub-tasks."""
        sub_tasks = self._decompose_question(question)

        # Pick the next unfinished task by checking what we've already done
        completed_types = {s.action for s in self.history if s.action}

        for task in sub_tasks:
            ttype = task["type"]
            if ttype == "calculate" and "calculate" not in completed_types:
                return "calculate", task["target"]
            if ttype == "retrieve" and "retrieve_documents" not in completed_types:
                return "retrieve_documents", question

        # No pending tasks — synthesise
        return "retrieve_documents", question

    def _execute_action(self, name: str, inp: str) -> str:
        tool = self.tools.get(name)
        if not tool:
            return f"Error: unknown tool '{name}'"
        try:
            return tool.fn(inp)
        except Exception as e:
            return f"Error executing {name}: {e}"

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    def run(self, question: str) -> str:
        """Run the ReAct loop to answer a question."""
        print(f"\nQuestion: {question}")
        print("-" * 50)
        self.history = []

        sub_tasks = self._decompose_question(question)
        print(f"  Decomposed into {len(sub_tasks)} sub-task(s): "
              f"{[t['type'] for t in sub_tasks]}")

        for step in range(self.max_steps):
            last_obs = self.history[-1].observation if self.history else ""

            # --- THINK ---
            if step == 0:
                thought = f"I need to answer: {question}"
                if len(sub_tasks) > 1:
                    thought += ". This has multiple parts. Let me start with the first part."
                else:
                    thought += ". Let me start by retrieving relevant documents."
            else:
                obs = (last_obs or "").lower()
                if "result:" in obs:
                    thought = "Good, I got the calculation result. "
                    thought += "Now let me check if there's retrieval information to combine."
                elif "no documents" in obs:
                    thought = "The knowledge base didn't have this. Let me try searching the web instead."
                else:
                    thought = "I have enough information now. Let me synthesize the final answer."

            print(f"\n[Step {step + 1}]")
            print(f"  Thought: {thought}")

            # --- ACT ---
            action_name, action_input = self._select_action(step, question, last_obs)
            print(f"  Action: {action_name}")
            print(f"  Action Input: \"{action_input}\"")

            # --- OBSERVE ---
            observation = self._execute_action(action_name, action_input)
            obs_trunc = observation[:120] + "..." if len(observation) > 120 else observation
            print(f"  Observation: {obs_trunc}")

            step_record = ReActStep(
                thought=thought,
                action=action_name,
                action_input=action_input,
                observation=observation,
            )
            self.history.append(step_record)

            # --- Decide if we can answer ---
            completed_types = {s.action for s in self.history if s.action}
            required_types = {t["type"] for t in sub_tasks}
            # Map retrieve task to actual tool name
            required_tools = set()
            for t in sub_tasks:
                if t["type"] == "retrieve":
                    required_tools.add("retrieve_documents")
                elif t["type"] == "calculate":
                    required_tools.add("calculate")
            if required_tools.issubset(completed_types):
                break

        # --- FINAL ANSWER ---
        final = self._generate_final_answer(question)
        print("\n" + "=" * 50)
        print("FINAL ANSWER:")
        print(final)
        return final

    def _generate_final_answer(self, question: str) -> str:
        """Synthesize the final answer from all steps."""
        observations = []
        for s in self.history:
            if s.observation and "Error" not in s.observation:
                observations.append(s.observation)

        answer = (
            f"Based on {len(self.history)} reasoning steps:\n\n"
            f"{question}\n\n"
        )
        if observations:
            answer += "Key findings:\n"
            for i, obs in enumerate(observations, 1):
                snippet = obs[:150] + "..." if len(obs) > 150 else obs
                answer += f"  {i}. {snippet}\n"

        answer += (
            f"\nThe agent used {len(self.history)} tool call(s) "
            f"({', '.join(s.action or 'thinking' for s in self.history)}) "
            f"to arrive at this answer through iterative reasoning."
        )
        return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Demo 6: Agentic RAG — ReAct Agent")
    print("=" * 60)

    agent = ReActAgent(TOOLS)

    # Question 1: Simple retrieval
    print("\n" + "#" * 50)
    print("# Query 1: Document Retrieval")
    print("#" * 50)
    agent.run("What is hybrid search and how does it work?")

    # Question 2: Multi-step with calculation
    print("\n" + "#" * 50)
    print("# Query 2: Multi-Step (Retrieve + Calculate)")
    print("#" * 50)
    agent.run("What is BM25 and calculate 15 + 27?")

    print("\n" + "=" * 60)
    print("ReAct agent demonstrates multi-step reasoning with tools.")
    print("=" * 60)


if __name__ == "__main__":
    main()
