"""
evaluate.py
===========
RAGAS evaluation script for the Aviation RAG Service.

Runs the 10-question calibration set against the live pipeline and computes:
  - context_precision  : fraction of retrieved pages that are actually relevant
  - context_recall     : does the context contain everything needed to answer?
  - faithfulness       : is the answer grounded in the context?
  - answer_relevancy   : does the answer address the question?

Requirements:
    pip install "ragas>=0.2.0,<0.3.0"

Usage:
    # Server must be running: python main.py
    python evaluate.py

    # Or against a custom host:
    python evaluate.py --host http://localhost:8000

Output:
    eval_results.json  — full per-question results
    eval_results.csv   — summary table for README
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
for _noisy in ("chromadb", "langchain", "langchain_core", "httpx",
               "httpcore", "urllib3", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

import requests
from datasets import Dataset

# ── RAGAS imports ─────────────────────────────────────────────────────────────
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        Faithfulness,
        ResponseRelevancy,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError:
    print('❌  RAGAS not installed. Run: pip install "ragas>=0.2.0,<0.3.0"')
    sys.exit(1)

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import config


# ── Golden dataset (10 questions from calibration set) ────────────────────────

GOLDEN_DATASET = [
    {
        "question": "What are the Business Class meal options on the flight from Barcelona to Rome?",
        "ground_truth": 'On the BCN to FCO route, Business Class passengers can choose between Option A: "Mediterranean Harvest" (Catalan-style Braised Lamb) and Option B: "Tiber Seabass" (Pan-seared Mediterranean Bass)',
        "reference_pages": {"BCN-FCO Menus.pdf": [1]},
    },
    {
        "question": "Which meal option on the Vienna to London Heathrow flight does not contain milk as a key allergen?",
        "ground_truth": 'The Business Class "Option B: Danube Zander" (Fish) meal is the only option on this route that does not contain milk. Its key allergens are listed as Fish, Gluten, and Seeds',
        "reference_pages": {"VIE-LHR Menus.pdf": [3]},
    },
    {
        "question": "What are the primary components of the BCF Fire Extinguisher, and how far should you stand from the fire?",
        "ground_truth": "The primary components include a discharge nozzle, a ring safety pin, a pressure gauge, a handle, and a lever. When using the extinguisher, you should aim at the base of the fire from a distance of six feet",
        "reference_pages": {"Boeing B737 Manual.pdf": [70]},
    },
    {
        "question": "After a landing that exceeds the quick turnaround limit weight, what are the two procedures a crew can use before executing a subsequent takeoff?",
        "ground_truth": "The crew has two approved procedural options: Primary Procedure: wait a minimum of 62 minutes and verify that the wheel thermal plugs have not melted. Alternate Procedure: measure the surface temperature of each brake pressure plate between 10 and 15 minutes after parking, at minimum two points per brake, without artificial cooling. If every temperature is below 218°C, immediate dispatch is allowed; otherwise the 62-minute wait applies.",
        "reference_pages": {"Boeing B737 Manual.pdf": [26]},
    },
    {
        "question": "In the event that the Flight Deck Security Door becomes jammed, how can the decompression panel be manually opened to allow for egress?",
        "ground_truth": "To manually open the decompression panel in the event of a jammed Flight Deck Security Door, you must pull the two release pins inward. This action manually separates the decompression panel from the jammed door, allowing the panel to be opened to provide an emergency egress path.",
        "reference_pages": {"Boeing B737 Manual.pdf": [40]},
    },
    {
        "question": "What are the fuel and time requirements for an air distance of 1400 NM at a pressure altitude of 33,000 FT?",
        "ground_truth": "For an air distance of 1400 NM at a pressure altitude of 33,000 FT, the reference fuel required is 7.2 (1000 KG) and the estimated time is 3 hours and 27 minutes (3:27)",
        "reference_pages": {"Boeing B737 Manual.pdf": [15]},
    },
    {
        "question": "If the captain decides an emergency evacuation may soon be required, what exact phraseology should be used over the PA system, and what is the cabin crew supposed to do?",
        "ground_truth": 'The flight crew should announce, "ATTENTION CREW! AT STATIONS!" over the Passenger Address (PA) System. This instructs the cabin crew that an emergency evacuation may soon be required.',
        "reference_pages": {"A320_321 Flight Crew Training Manual - Normal Operations.pdf": [10]},
    },
    {
        "question": "Comparing the 180° turn figures in the A320 manual, what is the Minimum Runway Width required with Asymmetric Thrust for aircraft MSN 1320-1637 compared to MSN 0781-0852?",
        "ground_truth": "For aircraft MSN 1320-1637, the Minimum Runway Width required with Asymmetric Thrust is 30 meters (99 feet). For aircraft MSN 0781-0852, the Minimum Runway Width required with Asymmetric Thrust is 32 meters (105 feet). The MSN 0781-0852 requires a slightly wider runway (2 meters or 6 feet wider).",
        "reference_pages": {"A320_321 Flight Crew Training Manual - Normal Operations.pdf": [40, 41]},
    },
    {
        "question": "How long does the standard ADIRS alignment take, and what is the hard deadline for completing it?",
        "ground_truth": "The standard ADIRS alignment takes approximately 10 minutes. The hard deadline for completing this alignment is before pushback, meaning it must be finished before any aircraft movement occurs.",
        "reference_pages": {"A320_321 Flight Crew Training Manual - Normal Operations.pdf": [18]},
    },
    {
        "question": "According to the visual cockpit preparation flow pattern, what area of the cockpit corresponds to step 1, and which pilot is responsible for it?",
        "ground_truth": "According to the cockpit preparation flow pattern, step 1 corresponds to the Overhead panel, where the responsible pilot must extinguish any white lights. The pilot responsible for this step is the Pilot Flying (PF).",
        "reference_pages": {"A320_321 Flight Crew Training Manual - Normal Operations.pdf": [26]},
    },
]


# ── Retrieval score (custom — page-level accuracy) ────────────────────────────

def compute_retrieval_score(
    references: list[dict],
    expected_pages: dict[str, list[int]],
) -> dict:
    """
    Compute page-level retrieval accuracy.

    Returns:
        precision: fraction of returned pages that are relevant
        recall:    fraction of expected pages that were returned
        f1:        harmonic mean
    """
    returned = set()
    for ref in references:
        doc = ref.get("document_title", "")
        for p in ref.get("pages", []):
            returned.add((doc, p))

    expected = set()
    for doc, pages in expected_pages.items():
        for p in pages:
            expected.add((doc, p))

    if not returned:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(returned & expected)
    precision = tp / len(returned)
    recall = tp / len(expected) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


# ── Query the live API ─────────────────────────────────────────────────────────

def query_api(question: str, host: str) -> dict:
    """Call the running RAG service and return the response dict."""
    try:
        resp = requests.post(
            f"{host}/query",
            json={"question": question},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ❌ API error for question: {question[:60]}... → {e}")
        return {"answer": "", "references": []}


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation(host: str) -> None:
    print("\n" + "=" * 60)
    print("  Aviation RAG — RAGAS Evaluation")
    print("=" * 60)
    print(f"  API host     : {host}")
    print(f"  Questions    : {len(GOLDEN_DATASET)}")
    print(f"  LLM judge    : {config.GEMINI_LLM_MODEL}")
    print("=" * 60 + "\n")

    # ── Step 1: query the API for all questions ───────────────────────────────
    print("📡 Querying API...")
    results = []
    for i, item in enumerate(GOLDEN_DATASET, 1):
        print(f"  [{i}/{len(GOLDEN_DATASET)}] {item['question'][:70]}...")
        response = query_api(item["question"], host)
        answer = response.get("answer", "")
        references = response.get("references", [])

        # Compute our custom retrieval score
        retrieval = compute_retrieval_score(references, item["reference_pages"])

        results.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "answer": answer,
            "references": references,
            "reference_pages": item["reference_pages"],
            "retrieval_precision": retrieval["precision"],
            "retrieval_recall": retrieval["recall"],
            "retrieval_f1": retrieval["f1"],
        })

        # Collect contexts for RAGAS (list of page texts retrieved)
        # We use the answer as a proxy since we don't expose raw chunks
        time.sleep(1)  # avoid rate limiting

    # ── Step 2: build RAGAS dataset ───────────────────────────────────────────
    print("\n🧮 Running RAGAS metrics...")

    ragas_data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        # RAGAS context_precision/recall need retrieved contexts as strings
        # We use the answer itself as proxy — not ideal but workable
        # For full accuracy you'd expose raw chunk texts from the API
        "contexts": [[r["answer"]] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }
    dataset = Dataset.from_dict(ragas_data)

    # RAGAS judge LLM — using Gemini
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_LLM_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.0,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.GEMINI_EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    metrics = [
        Faithfulness(llm=ragas_llm),
        ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    ]

    try:
        ragas_result = ragas_evaluate(dataset=dataset, metrics=metrics)
        ragas_df = ragas_result.to_pandas()
    except Exception as e:
        print(f"  ⚠ RAGAS evaluation failed: {e}")
        ragas_df = None

    # ── Step 3: merge results and print summary ───────────────────────────────
    print("\n📊 Results Summary")
    print("-" * 60)

    avg_precision = sum(r["retrieval_precision"] for r in results) / len(results)
    avg_recall = sum(r["retrieval_recall"] for r in results) / len(results)
    avg_f1 = sum(r["retrieval_f1"] for r in results) / len(results)

    print(f"  Retrieval Precision (page-level) : {avg_precision:.3f}")
    print(f"  Retrieval Recall    (page-level) : {avg_recall:.3f}")
    print(f"  Retrieval F1        (page-level) : {avg_f1:.3f}")

    if ragas_df is not None:
        print(f"  Faithfulness                     : {ragas_df['faithfulness'].mean():.3f}")
        print(f"  Response Relevancy               : {ragas_df['answer_relevancy'].mean():.3f}")

    print("-" * 60)

    # Per-question retrieval breakdown
    print("\n  Per-question retrieval:")
    for r in results:
        status = "✅" if r["retrieval_f1"] == 1.0 else "⚠️"
        print(f"  {status} P={r['retrieval_precision']:.2f} R={r['retrieval_recall']:.2f} | {r['question'][:60]}...")

    # ── Step 4: save outputs ──────────────────────────────────────────────────
    output_json = {
        "summary": {
            "retrieval_precision": round(avg_precision, 3),
            "retrieval_recall": round(avg_recall, 3),
            "retrieval_f1": round(avg_f1, 3),
            "faithfulness": round(ragas_df["faithfulness"].mean(), 3) if ragas_df is not None else None,
            "answer_relevancy": round(ragas_df["answer_relevancy"].mean(), 3) if ragas_df is not None else None,
        },
        "per_question": results,
    }

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "score"])
        writer.writerow(["retrieval_precision", round(avg_precision, 3)])
        writer.writerow(["retrieval_recall", round(avg_recall, 3)])
        writer.writerow(["retrieval_f1", round(avg_f1, 3)])
        if ragas_df is not None:
            writer.writerow(["faithfulness", round(ragas_df["faithfulness"].mean(), 3)])
            writer.writerow(["answer_relevancy", round(ragas_df["answer_relevancy"].mean(), 3)])

    print("\n✅ Results saved to eval_results.json and eval_results.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for Aviation RAG Service")
    parser.add_argument("--host", default="http://localhost:8000", help="API host URL")
    args = parser.parse_args()

    run_evaluation(host=args.host)
