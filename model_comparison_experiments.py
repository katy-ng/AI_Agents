"""Helpers for the controlled model comparison added to tutorial_1.ipynb."""

import hashlib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

from scrag.agents.ask import CONTRACT, TEMPERATURE, answer
from scrag.agents.tools import gate_run, retrieve_logged
from scrag.store.gate import citations_in


# Pin versions instead of using a changing "latest" alias.
MODELS = {
    "Haiku 4.5": "anthropic/claude-haiku-4.5",
    "Sonnet 4.6": "anthropic/claude-sonnet-4.6",
    "Opus 4.6": "anthropic/claude-opus-4.6",
    "Fable 5.1": "anthropic/claude-fable-5.1",
}

QUESTIONS = [
    {
        "id": "evidence-map",
        "kind": "complex synthesis",
        "question": (
            "Within the retrieved literature, what evidence links glial senescence "
            "to neurodegeneration in Drosophila? Name the cellular or molecular markers "
            "and readouts reported, distinguish direct observations from proposed mechanisms, "
            "and identify the study system for each observation."
        ),
    },
    {
        "id": "state-discrimination",
        "kind": "complex synthesis",
        "question": (
            "How do the retrieved studies distinguish cellular senescence from other stress-, "
            "inflammatory-, or aging-associated glial states in relation to neurodegeneration? "
            "Compare the markers, model systems, and limitations of the evidence without "
            "treating association as causation."
        ),
    },
    {
        "id": "causality",
        "kind": "complex synthesis",
        "question": (
            "What evidence in the retrieved literature supports or contradicts a causal role "
            "for senescent glia in neurodegeneration? Separate interventions from observational "
            "associations and state where the chunks cannot establish causation."
        ),
    },
    {
        "id": "false-premise",
        "kind": "abstention stress test",
        "question": (
            "Which conserved master regulator causes glial senescence and neurodegeneration "
            "across mammals and flies, and which intervention proves that it is sufficient? "
            "If the retrieved chunks do not establish this premise, say so rather than "
            "supplying an answer from outside them."
        ),
    },
]


def configure(root, models=None, questions=None, k=20, repeats=1, corpus="hypothesis"):
    """Make an explicit, reproducible configuration."""
    root = Path(root)
    store = Path(os.environ.get("SCRAG_STORE", root / "data"))
    if not (store / "store.sqlite").is_file() or not (store / "vectors.npy").is_file():
        raise FileNotFoundError(
            f"Extract students-v1.zip and set SCRAG_STORE to its folder; expected store.sqlite and vectors.npy in {store}"
        )
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise EnvironmentError("Set OPENROUTER_API_KEY before running paid model calls.")
    if TEMPERATURE != 0:
        raise RuntimeError(f"Expected TEMPERATURE=0, found {TEMPERATURE}.")
    if k < 1 or repeats < 1:
        raise ValueError("k and repeats must be positive.")
    return {
        "root": root,
        "store": store,
        "models": dict(models or MODELS),
        "questions": list(questions or QUESTIONS),
        "k": k,
        "repeats": repeats,
        "corpus": corpus,
        "run_dir": root / "reports" / "runs" / "model_comparison" / time.strftime("%Y-%m-%d_%H%M%S"),
    }


def _hash(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def run_model_comparison(config):
    """Run one shared retrieval per question, then replay it to each model."""
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {key: config[key] for key in ("models", "questions", "k", "repeats", "corpus")}
    manifest.update(temperature=TEMPERATURE, store=str(config["store"]), contract=CONTRACT, retrieval_hashes={})
    evidence = {}
    for item in config["questions"]:
        retrieval_dir = run_dir / item["id"] / "retrieval"
        records = retrieve_logged(retrieval_dir, item["question"], corpus=config["corpus"], k=config["k"])
        (retrieval_dir / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        evidence[item["id"]] = records
        manifest["retrieval_hashes"][item["id"]] = _hash(records)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for item in config["questions"]:
        retrieval_dir = run_dir / item["id"] / "retrieval"
        retrieval_log = (retrieval_dir / "retrieved.jsonl").read_text(encoding="utf-8")
        for repeat in range(1, config["repeats"] + 1):
            for label, model in config["models"].items():
                trial = run_dir / item["id"] / f"repeat-{repeat:02d}" / label.lower().replace(" ", "-")
                trial.mkdir(parents=True)
                # The copied log lets the citation gate judge precisely the shared chunks.
                (trial / "retrieved.jsonl").write_text(retrieval_log, encoding="utf-8")
                started = time.perf_counter()
                row = {
                    "question_id": item["id"], "question_kind": item["kind"], "question": item["question"],
                    "repeat": repeat, "model_label": label, "model_requested": model,
                    "retrieval_hash": manifest["retrieval_hashes"][item["id"]],
                    "retrieved_chunks": len(evidence[item["id"]]), "run_dir": str(trial),
                    "records_path": str(retrieval_dir / "records.json"),
                }
                try:
                    text, cost, served = answer(
                        item["question"], evidence[item["id"]], model=model, system=CONTRACT, record_to=trial
                    )
                    (trial / "answer.md").write_text(text, encoding="utf-8")
                    report = gate_run(trial, text)
                    row.update(
                        answer=text, answer_path=str(trial / "answer.md"), model_served=served,
                        cost_usd=cost, latency_seconds=round(time.perf_counter() - started, 2),
                        gate_pass=report["pass"], gate_failed=report["failed"], gate_rows=report["citations"], error=""
                    )
                except Exception as exc:
                    (trial / "error.txt").write_text(repr(exc), encoding="utf-8")
                    row.update(
                        answer="", answer_path="", model_served="", cost_usd=float("nan"),
                        latency_seconds=round(time.perf_counter() - started, 2),
                        gate_pass=False, gate_failed=0, gate_rows=[], error=repr(exc)
                    )
                rows.append(row)
                print(item["id"], repeat, label, "ERROR" if row["error"] else ("pass" if row["gate_pass"] else "GATE FAIL"))
    (run_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return pd.DataFrame(rows)


def _sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _ends_with_citation(sentence):
    found = citations_in(sentence)
    return bool(found and re.fullmatch(r"\)?[.!?]*", sentence[found[-1]["end"]:].strip()))


def score_outputs(results):
    """Measure verbosity and automatable portions of the answer contract."""
    rows = []
    for _, result in results.iterrows():
        text = result["answer"] if not result["error"] else ""
        sentences, citations = _sentences(text), citations_in(text) if text else []
        terminal = sum(_ends_with_citation(sentence) for sentence in sentences)
        plain = not any(re.match(r"\s*(#|[-*+]\s|\d+\.\s|\|)", line) for line in text.splitlines())
        refusal = len(citations) == 0 and len(sentences) == 1
        all_cited = refusal or (bool(sentences) and terminal == len(sentences))
        rows.append({
            "word_count": len(re.findall(r"\b[\w'-]+\b", text)),
            "character_count": len(text), "sentence_count": len(sentences), "citation_count": len(citations),
            "invalid_citation_count": int(result["gate_failed"]), "citation_gate_pass": bool(result["gate_pass"]),
            "five_sentence_limit": len(sentences) <= 5, "plain_prose_proxy": plain,
            "all_sentences_terminally_cited_or_refusal": all_cited,
            "contract_proxy_pass": bool(not result["error"] and result["gate_pass"] and len(sentences) <= 5 and plain and all_cited),
        })
    return pd.concat([results.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def show_summary(scored):
    """Show comparison results. Matching hashes confirm retrieval was controlled."""
    if scored.empty:
        return None
    run_dir = Path(scored["run_dir"].iloc[0]).parents[2]
    scored.to_csv(run_dir / "scored_results.csv", index=False)
    good = scored[scored["error"].eq("")].copy()
    if good.empty:
        print("No model calls completed; inspect error.txt files.")
        return None
    assert good.groupby("question_id")["retrieval_hash"].nunique().max() == 1, "retrieval drift detected"
    summary = good.groupby("model_label", as_index=False).agg(
        runs=("model_label", "size"), mean_words=("word_count", "mean"),
        median_words=("word_count", "median"), mean_sentences=("sentence_count", "mean"),
        gate_pass_rate=("citation_gate_pass", "mean"),
        contract_proxy_pass_rate=("contract_proxy_pass", "mean"),
        mean_invalid_citations=("invalid_citation_count", "mean"), total_cost_usd=("cost_usd", "sum")
    ).sort_values("mean_words")
    display(Markdown("### Experiment 1: controlled complex-RAG comparison"))
    display(Markdown("Tests whether output length and citation or contract failures differ by model when the evidence, question, contract, temperature, and k are held fixed."))
    display(summary.round(3))
    display(good[[
        "question_id", "question_kind", "repeat", "model_label", "retrieval_hash",
        "word_count", "sentence_count", "citation_count", "invalid_citation_count",
        "citation_gate_pass", "contract_proxy_pass", "cost_usd", "answer_path"
    ]].sort_values(["question_id", "repeat", "model_label"]))
    return summary


def make_semantic_audit_template(scored):
    """Make a human audit for claims that valid citations may still not support."""
    rows = []
    for _, result in scored[scored["error"].eq("")].iterrows():
        for number, sentence in enumerate(_sentences(result["answer"]), 1):
            cited = citations_in(sentence)
            rows.append({
                "question_id": result["question_id"], "model_label": result["model_label"],
                "repeat": result["repeat"], "sentence_number": number, "sentence": sentence,
                "has_citation": bool(cited), "cited_identifiers": "; ".join(item["text"] for item in cited),
                "citation_gate_pass_for_answer": result["citation_gate_pass"],
                "is_literature_finding": "", "support": "", "auditor_notes": "",
                "records_path": result["records_path"], "answer_path": result["answer_path"],
            })
    audit = pd.DataFrame(rows)
    path = Path(scored["run_dir"].iloc[0]).parents[2] / "semantic_audit_template.csv"
    audit.to_csv(path, index=False)
    display(Markdown("### Experiment 2: sentence-level semantic grounding audit"))
    display(Markdown("Mark literature-finding sentences and whether the saved retrieved chunks support them. This measures hallucination or overstatement that a syntactically valid citation gate cannot detect. Save the completed file as semantic_audit_completed.csv in the same folder."))
    display(audit.head(12))
    print(path)
    return path


def summarize_completed_audit(scored, filename="semantic_audit_completed.csv"):
    """Calculate unsupported-claim rates after a human completes the audit."""
    run_dir = Path(scored["run_dir"].iloc[0]).parents[2]
    path = run_dir / filename
    if not path.exists():
        print("Complete the semantic audit template first.")
        return None
    audit = pd.read_csv(path, keep_default_na=False)
    findings = audit[audit["is_literature_finding"].str.casefold().eq("yes")].copy()
    findings["support"] = findings["support"].str.casefold().str.strip()
    findings = findings[findings["support"].isin(["yes", "no", "unclear"])]
    summary = findings.assign(
        supported=findings["support"].eq("yes"), unsupported=findings["support"].eq("no"),
        unclear=findings["support"].eq("unclear")
    ).groupby("model_label", as_index=False)[["supported", "unsupported", "unclear"]].sum()
    summary["audited_findings"] = summary[["supported", "unsupported", "unclear"]].sum(axis=1)
    summary["unsupported_claim_rate"] = summary["unsupported"] / summary["audited_findings"]
    summary.to_csv(run_dir / "semantic_audit_summary.csv", index=False)
    display(summary.round(3))
    return summary
