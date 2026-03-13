# evaluate_agent.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langsmith import Client
from langsmith.evaluation import evaluate, EvaluationResult
from langsmith.schemas import Run, Example
from langchain_openai import ChatOpenAI


from src.agent.rag_agent import agent_executor


# ---------------- WRAPPER AGENT ----------------

def run_agent(inputs: dict) -> dict:
    result = agent_executor.invoke({"input": inputs["input"]})
    return {
        "output": result.get("output", ""),
        "intermediate_steps": result.get("intermediate_steps", [])
    }


# ---------------- LOAD DATASET ----------------

def ensure_dataset(name: str):
    client = Client()

    try:
        dataset = client.read_dataset(dataset_name=name)
        print(f"[INFO] Dataset '{name}' found.")
    except Exception:
        print(f"[INFO] Dataset '{name}' not found. Creating it now...")

        dataset = client.create_dataset(
            dataset_name=name,
            description="Auto-created test dataset for RAG + tools agent."
        )

        default_examples = [
            {
                "input": "Ce spune Sima despre cotatia petrolului?",
                "expected_output": (
                    "Petrolul a avut o zi extrem de volatilă: a urcat de la 90 la 119 dolari, apoi a căzut " \
                    "la 83. Contractele futures au rămas la ~68 dolari, semn că explozia prețului spot a fost " \
                    "speculativă, nu o schimbare reală de trend. Pe termen mediu, dacă războiul continuă, " \
                    "prețul s-ar putea stabiliza la 100–110 dolari, dar scenariile de 200–300 dolari sunt " \
                    "considerate improbabile datorită capacității SUA/Canada de a crește rapid producția. " \
                    "În România, chiar fără importuri din Golf, benzina și inflația vor crește deoarece " \
                    "companiile vor profita de context pentru a vinde mai scump. "
                )
            },
            {
                "input": "Care este tickerul pentru firma Tesla?",
                "expected_output": "Tickerul pentru Tesla este TSLA."
            },
            {
                "input": "Câte ore are un pinguin?",
                "expected_output": "Nu știu."
            }
        ]

        for ex in default_examples:
            client.create_example(
                dataset_id=dataset.id,
                inputs={"input": ex["input"]},
                outputs={"expected_output": ex["expected_output"]}
            )

        print(f"[INFO] Dataset '{name}' created with {len(default_examples)} examples.")

    examples = list(client.list_examples(dataset_id=dataset.id))
    print(f"[INFO] Loaded {len(examples)} examples from dataset '{name}'.")
    return examples


# ---------------- BUILT-IN EVALUATORS ----------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def _llm_score(prompt: str) -> float:
    """Send a prompt to the LLM and give a score between 1-10."""
    response = llm.invoke(prompt)
    text = response.content.strip()
    
    print(f"[DEBUG] LLM raw response: '{text}'")

    for token in text.split():
        try:
            score = float(token.strip(".,"))
            if 1 <= score <= 10:
                print(f"[DEBUG] Score found: {score}")
                return score / 10
        except ValueError:
            continue
    print("[DEBUG] Fallback to 0.5")
    return 0.5  # fallback


def correctness_evaluator(run: Run, example: Example) -> EvaluationResult:
    prediction = run.outputs.get("output", "") if run.outputs else ""
    reference  = example.outputs.get("expected_output", "") if example.outputs else ""
    question   = example.inputs.get("input", "")
    prompt = (
        f"You are a grader. Given a question, a reference answer, and a prediction, "
        f"score the prediction from 1 to 10 based on factual correctness.\n\n"
        f"Question: {question}\n"
        f"Reference: {reference}\n"
        f"Prediction: {prediction}\n\n"
        f"Respond with ONLY a single digit number, nothing else."
    )
    return EvaluationResult(key="correctness", score=_llm_score(prompt))


def relevance_evaluator(run: Run, example: Example) -> EvaluationResult:
    prediction = run.outputs.get("output", "") if run.outputs else ""
    question   = example.inputs.get("input", "")
    prompt = (
        f"You are a grader. Score the relevance of the answer to the question from 1 to 10.\n\n"
        f"Question: {question}\n"
        f"Answer: {prediction}\n\n"
        f"Respond with ONLY a single digit number, nothing else."
    )
    return EvaluationResult(key="relevance", score=_llm_score(prompt))


def groundedness_evaluator(run: Run, example: Example) -> EvaluationResult:
    prediction = run.outputs.get("output", "") if run.outputs else ""
    question   = example.inputs.get("input", "")
    prompt = (
        f"You are a grader. Score how grounded and factual the answer is (not hallucinated) from 1 to 10.\n\n"
        f"Question: {question}\n"
        f"Answer: {prediction}\n\n"
        f"Respond with ONLY a single digit number, nothing else."
    )
    return EvaluationResult(key="groundedness", score=_llm_score(prompt))


# ---------------- CUSTOM TOOL USE EVALUATOR ----------------

def tool_use_evaluator(run: Run, example: Example) -> EvaluationResult:
    steps = run.outputs.get("intermediate_steps", []) if run.outputs else []
    score = 1.0
    comments = []

    for action, observation in steps:
        if action.tool == "rag_qa" and not isinstance(action.tool_input, str):
            score -= 0.3
            comments.append("rag_qa input should be a string")
        if action.tool == "get_stock_ticker" and not isinstance(action.tool_input, str):
            score -= 0.3
            comments.append("get_stock_ticker input should be a string")

    return EvaluationResult(
        key="tool_use",
        score=max(score, 0),
        comment="\n".join(comments)
    )


def run_builtin_evals():
    ensure_dataset("rag-agent-tests")

    evaluate(
        run_agent,
        data="rag-agent-tests",
        evaluators=[correctness_evaluator, relevance_evaluator, groundedness_evaluator]
    )


def run_tool_eval():
    ensure_dataset("rag-agent-tests")

    evaluate(
        run_agent,
        data="rag-agent-tests",
        evaluators=[tool_use_evaluator]
    )


def evaluate_live(question: str, answer: str, steps: list) -> dict:
    """Evalueaza un raspuns in timp real: relevance, groundedness, tool_use."""

    relevance_prompt = (
        f"You are a grader. Score the relevance of the answer to the question from 1 to 10.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Respond with ONLY a single digit number, nothing else."
    )

    groundedness_prompt = (
        f"You are a grader. Score how grounded and factual the answer is (not hallucinated) from 1 to 10.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Respond with ONLY a single digit number, nothing else."
    )

    tool_score = 1.0
    for action, observation in steps:
        if hasattr(action, "tool"):
            if action.tool == "rag_qa" and not isinstance(action.tool_input, str):
                tool_score -= 0.3
            if action.tool == "get_stock_ticker" and not isinstance(action.tool_input, str):
                tool_score -= 0.3

    return {
        "relevance":    round(_llm_score(relevance_prompt), 2),
        "groundedness": round(_llm_score(groundedness_prompt), 2),
        "tool_use":     round(max(tool_score, 0), 2),
    }


# ---------------- MAIN ----------------

if __name__ == "__main__":
    run_builtin_evals()
    run_tool_eval()