from __future__ import annotations

from typing import Any, Dict, List


def _load_ragas():
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        return evaluate, answer_relevancy, context_precision, faithfulness
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ragas is not available in the current environment. "
            "Reinstall dependencies after aligning langchain-community with ragas, e.g.: "
            "pip install -U 'langchain-community>=0.4.2' 'ragas>=0.2.15'"
        ) from exc


def evaluate_rag(
    question: str, answer: str, contexts: List[str], ground_truth: str | None = None
) -> Dict[str, Any]:
    """Evaluate a single RAG answer with Ragas metrics when possible."""
    if not question or not answer:
        return {
            "status": "skipped",
            "reason": "question and answer are required",
        }

    if not contexts:
        return {
            "status": "skipped",
            "reason": "no context provided for evaluation",
        }

    evaluate, answer_relevancy, context_precision, faithfulness = _load_ragas()

    metrics = [context_precision, faithfulness]
    if ground_truth:
        metrics.append(answer_relevancy)

    data = [
        {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": ground_truth,
        }
    ]

    result = evaluate(data, metrics=metrics)
    scores = {}
    for name, value in result.items():
        if isinstance(value, dict):
            for metric_name, metric_value in value.items():
                scores[metric_name] = float(metric_value)
        else:
            scores[name] = float(value)

    return {
        "status": "ok",
        "scores": scores,
    }
