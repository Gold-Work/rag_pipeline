"""RAGAS evaluation script for the RAG pipeline.

Evaluates three quality metrics against a golden dataset of 10 questions:
  - faithfulness        : answer grounded in retrieved context
  - answer_relevancy    : answer addresses the question
  - context_precision   : retrieved context contains relevant information

Quality gate: each metric must be >= THRESHOLD (0.75).

Usage:
    python scripts/eval_pipeline.py

Exits 0 if all metrics pass or if ChromaDB is empty (no data to evaluate).
Exits 1 if any metric is below the threshold.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

from src.query.augmenter import build_prompt
from src.query.generator import generate
from src.query.reranker import rerank
from src.query.retriever import retrieve
from src.utils.config import get_config

config = get_config()
THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Golden dataset — 10 questions covering all 4 indexed documents
# ---------------------------------------------------------------------------

GOLDEN_DATASET = [
    # IBM Policy Assessment & Compliance
    (
        "What is the scope of IBM's policy compliance assessment program?",
        "Covers all business units and geographies to ensure adherence to internal policies and external regulations.",
    ),
    (
        "What types of controls does IBM use in its compliance program?",
        "IBM uses preventive, detective, and corrective controls.",
    ),
    # IBM Unstructured Data Identification & Management
    (
        "What qualifies as unstructured data according to IBM?",
        "Emails, documents, images, audio, video, and other content without a predefined schema.",
    ),
    (
        "How should sensitive unstructured data be classified and protected at IBM?",
        "Classified, labeled, and protected with access controls limiting exposure per IBM data governance policies.",
    ),
    # Practical Guide to Building Agents
    (
        "What are the main components of an AI agent?",
        "Perception/input processing, reasoning/planning, and action/tool-use modules.",
    ),
    (
        "What is the role of tools in an AI agent system?",
        "Tools allow agents to interact with external systems, retrieve information, and perform actions"
        " beyond text generation.",
    ),
    (
        "How does an orchestrator differ from a subagent in multi-agent systems?",
        "An orchestrator coordinates subagents and delegates tasks; subagents handle specific subtasks.",
    ),
    # Understanding Machine Learning Theory and Algorithms
    (
        "What is the difference between supervised and unsupervised learning?",
        "Supervised learning uses labeled data to learn input-output mappings;"
        " unsupervised learning finds patterns in unlabeled data.",
    ),
    (
        "How does gradient descent update model parameters?",
        "Moves parameters opposite to the loss gradient direction, scaled by a learning rate.",
    ),
    (
        "What is overfitting and how can it be prevented?",
        "Overfitting is when a model memorizes training data and fails to generalize;"
        " prevented by regularization, cross-validation, and more data.",
    ),
]


def main() -> None:
    # Guard: skip evaluation if no documents are indexed
    client = chromadb.PersistentClient(path=config["paths"]["chroma_db"])
    collection = client.get_or_create_collection(config["embedding"]["collection_name"])
    if collection.count() == 0:
        print("Warning: ChromaDB is empty — evaluation skipped (no indexed documents)")
        sys.exit(0)

    print(f"=== RAGAS Evaluation — {len(GOLDEN_DATASET)} questions ===\n")

    # RAGAS picks up OPENAI_API_KEY from environment automatically
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]

    samples = []
    top_k_rerank = config["retrieval"]["top_k_rerank"]

    for i, (question, reference) in enumerate(GOLDEN_DATASET, 1):
        print(f"  [{i}/{len(GOLDEN_DATASET)}] {question[:70]}...")
        docs, _ = retrieve(question, k=20)
        docs = rerank(question, docs, top_k=top_k_rerank)
        prompt = build_prompt(question, docs)
        answer = generate(prompt)
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=[d.page_content for d in docs],
                reference=reference,
            )
        )

    print()
    result = evaluate(EvaluationDataset(samples=samples), metrics=metrics)

    print("\n=== Résultats ===")
    failed = False
    for metric_name, score in result.items():
        status = "✓" if score >= THRESHOLD else "✗"
        print(f"  {metric_name:<25}: {score:.3f}  {status}  (seuil {THRESHOLD})")
        if score < THRESHOLD:
            failed = True

    if failed:
        print("\n❌ Qualité insuffisante — au moins une métrique sous le seuil.")
        sys.exit(1)
    else:
        print("\n✅ Toutes les métriques passent le seuil.")
        sys.exit(0)


if __name__ == "__main__":
    main()
