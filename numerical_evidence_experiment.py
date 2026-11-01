"""Controlled synthetic-data / frozen-literature experiment for tutorial_1.ipynb.

Only run_experiment(..., dry_run=False) sends model requests. Dry runs retrieve
once per question from students-v1, curate the evidence, and save the plan.
"""

import csv
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from IPython.display import display

from scrag.agents.ask import CONTRACT, ENDPOINT, RETRIES, TEMPERATURE, TRANSIENT
from scrag.agents.tools import gate_run, retrieve_logged
from scrag.store.gate import citations_in

MODELS = {
    "Haiku 4.5": "anthropic/claude-haiku-4.5",
    "Sonnet 4.6": "anthropic/claude-sonnet-4.6",
    "Opus 4.6": "anthropic/claude-opus-4.6",
}
SEED = 20261020
MAX_TOKENS = 768  # five prose sentences; far below provider default of 65,536
INTER_CALL_DELAY_SECONDS = 20
CANDIDATE_K = 20
ROW_CITE = re.compile(r"\(DATA:([A-Z0-9-]+)\)")
SENTENCES = re.compile(r"(?<=[.!?])\s+")

# Values are synthetic experimental inputs, never observations from the papers.
# Every pair changes one numerical field in one row; all metadata and other rows match.
QUESTIONS = [
    {
        "id": "relish", "gene": "Relish", "target_id": "REL-01",
        "comparison": "aged glia vs young glia", "control_gene": "GstD1",
        "question": "Report the synthetic Relish and GstD1 results for aged glia vs young glia, then relate them cautiously to the supplied fly glial-senescence literature. What remains unsupported?",
        "retrieval_question": "For synthetic aged glia vs young glia, what are Relish's log2 fold change, adjusted p-value, direction, and significance; how does this compare with unchanged GstD1, and what can the retrieved literature support about glial senescence without implying that this synthetic expression contrast proves a mechanism?",
        "a": 1.4, "b": -1.4, "control": 0.2,
    },
    {
        "id": "atg8a", "gene": "Atg8a", "target_id": "ATG-01",
        "comparison": "older fly glia vs younger fly glia", "control_gene": "Ref(2)P",
        "question": "Report the synthetic Atg8a and Ref(2)P results for older fly glia vs younger fly glia, then relate them cautiously to the supplied fly brain and glial-aging literature. What remains unsupported?",
        "retrieval_question": "For synthetic older fly glia vs younger fly glia, what are Atg8a's log2 fold change, adjusted p-value, direction, and significance; how does this compare with unchanged Ref(2)P, and what does the retrieved literature permit one to infer about glial aging without claiming that this synthetic differential expression establishes causality?",
        "a": 1.6, "b": -1.6, "control": 0.3,
    },
]

# Selection is applied to one k=20 retrieval per question. The original ranked
# records remain saved; only these source-reviewed chunks are shown to models.
SELECTED_CHUNK_IDS = {
    "relish": ["10.1111/acel.13947#7", "10.1038/s41586-024-07516-8#12",
               "10.1371/journal.pbio.3002328#2", "10.1093/lifemeta/loae031#1"],
    "atg8a": ["10.1111/acel.13947#7", "10.1111/acel.13947#11",
               "10.1111/acel.13559#7", "10.1038/s41586-024-07516-8#6",
               "10.1038/s41586-024-07516-8#12"],
}

# Human-reviewed narrow claims, excluded from model messages. The excerpts are
# exact text anchors; substring validation is not a semantic support judgment.
LITERATURE_CLAIMS = {
    "relish": [
        {"claim": "In aged fly brains, glial AP1 blockade lowered senescence biomarkers while mitochondrial-status markers were unchanged.",
         "chunk_id": "10.1038/s41586-024-07516-8#12",
         "excerpt": "AP1-target genes decreased (Extended Data Fig. 7b), as did senescence biomarkers (Extended Data Fig. 7c). However, markers of mitochondrial status—inner complex gene expression and brain ATP levels—were unchanged"},
        {"claim": "A secondary account reports regionally progressive AP1-positive glia and senescence-associated features in aging flies.",
         "chunk_id": "10.1093/lifemeta/loae031#1",
         "excerpt": "AP1+ glia accumulated in a regionally progressive manner as the brain ages."},
    ],
    "atg8a": [
        {"claim": "Processed Atg8a increased with age in dissected fly brains; this is not a glia-specific expression result.",
         "chunk_id": "10.1111/acel.13559#7",
         "excerpt": "We found that processed Atg8a levels progressively increased normally in the brain with age"},
        {"claim": "In aged fly brains, glial AP1 blockade lowered senescence biomarkers while mitochondrial-status markers were unchanged.",
         "chunk_id": "10.1038/s41586-024-07516-8#12",
         "excerpt": "AP1-target genes decreased (Extended Data Fig. 7b), as did senescence biomarkers (Extended Data Fig. 7c). However, markers of mitochondrial status—inner complex gene expression and brain ATP levels—were unchanged"},
    ],
}

EXTENSION = """The added dataset is synthetic, not a paper finding. Cite its rows as (DATA:<row_id>); keep literature citations exactly as in the original contract. Do not cross-attribute dataset values and paper findings. Sentence 1: In [comparison], [gene] has log2 fold change [signed decimal] and adjusted p-value [decimal], indicating significant upregulation/downregulation (DATA:[target row ID]). Sentence 2: In the same comparison, [control gene] has log2 fold change [signed decimal] and adjusted p-value [decimal], indicating no significant change (DATA:[control row ID]). Then state only literature-supported context with exact chunk citations, or say it is unavailable. Still report dataset values if literature lacks an answer. Synthetic differential expression alone proves neither causality, senescence, nor gene function. Keep the original five-sentence plain-prose and quotation rules."""


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def datasets_for(item):
    shared = {"source": "synthetic experimental input; not a paper finding", "comparison": item["comparison"],
              "measure": "log2 fold change", "unit": "log2 ratio (dimensionless)",
              "adjusted_p_method": "synthetic adjusted p-value", "significance_rule": "adjusted p-value < 0.05",
              "direction_rule": "positive = upregulated; negative = downregulated, relative to denominator"}
    control = {"row_id": item["id"].upper() + "-02", "gene": item["control_gene"],
               "log2_fold_change": item["control"], "adjusted_p_value": 0.31}
    versions = {}
    for version, value in (("A", item["a"]), ("B", item["b"])):
        versions[version] = {**shared, "rows": [
            {"row_id": item["target_id"], "gene": item["gene"],
             "log2_fold_change": value, "adjusted_p_value": 0.01}, dict(control)]}
    return versions


def expected_for(item, version):
    value = item["a"] if version == "A" else item["b"]
    return {"target_row_id": item["target_id"], "comparison": item["comparison"],
            "gene": item["gene"], "log2_fold_change": value, "adjusted_p_value": 0.01,
            "significant": True, "direction": "upregulation" if value > 0 else "downregulation",
            "control_row_id": item["id"].upper() + "-02", "control_gene": item["control_gene"],
            "control_log2_fold_change": item["control"], "control_adjusted_p_value": 0.31,
            "control_significant": False,
            "interpretation_change": "target direction reverses between A and B; control remains nonsignificant",
            "supported_literature_claims": [claim["claim"] for claim in LITERATURE_CLAIMS[item["id"]]],
            "unsupported_conclusions": ["synthetic values are findings from cited papers",
                "differential expression proves causality or a senescent cell state",
                "a gene function or mechanism absent from the retrieved chunks"]}


def validate_design(items=QUESTIONS):
    assert len(items) == 2 and len({x["id"] for x in items}) == 2
    assert set(SELECTED_CHUNK_IDS) == {x["id"] for x in items}
    for item in items:
        a, b = datasets_for(item).values()
        assert a.keys() == b.keys() and a["rows"][1] == b["rows"][1]
        assert {k: v for k, v in a.items() if k != "rows"} == {k: v for k, v in b.items() if k != "rows"}
        differences = [(i, k) for i, (ra, rb) in enumerate(zip(a["rows"], b["rows"]))
                       for k in ra if ra[k] != rb[k]]
        assert differences == [(0, "log2_fold_change")]
        assert a["rows"][0]["log2_fold_change"] > 0 > b["rows"][0]["log2_fold_change"]
        assert all(x["adjusted_p_value"] < 0.05 for x in (a["rows"][0], b["rows"][0]))
        assert item["gene"] in item["question"] and item["comparison"] in item["question"]
        assert expected_for(item, "A")["direction"] != expected_for(item, "B")["direction"]
    return True


def configure(root, repeats=2, models=None, k=5, seed=SEED, corpus="hypothesis",
              max_tokens=MAX_TOKENS, batch_size=6, frozen_from=None):
    root = Path(root)
    store = Path(os.environ.get("SCRAG_STORE", root / "data"))
    if not all((store / name).is_file() for name in ("store.sqlite", "vectors.npy")):
        raise FileNotFoundError("students-v1 store needs store.sqlite and vectors.npy")
    if TEMPERATURE != 0 or repeats < 1 or k != 5 or max_tokens < 256 or batch_size < 1:
        raise ValueError("This experiment requires temperature=0, five selected chunks, max_tokens>=256, and positive repetitions/batch size")
    selected = dict(models or MODELS)
    if not selected or any(model not in MODELS.values() for model in selected.values()):
        raise ValueError("Use the pinned Haiku 4.5, Sonnet 4.6, and Opus 4.6 model IDs")
    validate_design()
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    return {"root": root, "store": store, "models": selected, "repeats": repeats,
            "k": k, "seed": seed, "corpus": corpus, "max_tokens": max_tokens,
            "batch_size": batch_size, "experiment_id": experiment_id,
            "frozen_from": str(frozen_from) if frozen_from else None,
            "run_dir": root / "reports" / "runs" / "numerical_evidence" / experiment_id}


def resume_config(run_dir):
    """Recreate the exact config for the next batch, even after a kernel restart."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return {"root": run_dir.parents[3], "store": Path(manifest["store"]),
            "models": manifest["models"], "repeats": manifest["repeats"],
            "k": manifest["k"], "seed": manifest["seed"], "corpus": manifest["corpus"],
            "max_tokens": manifest["max_tokens"], "batch_size": manifest["batch_size"],
            "frozen_from": manifest.get("frozen_from"),
            "experiment_id": manifest["experiment_id"], "run_dir": run_dir}


def _prompt(item, dataset):
    shown = {"source": "synthetic, not from papers", "comparison": dataset["comparison"],
             "significance_rule": dataset["significance_rule"], "rows": dataset["rows"]}
    return item["question"] + "\nSynthetic dataset: " + json.dumps(shown, ensure_ascii=False, separators=(",", ":"))


def _matrix(config):
    rows = [{"question_id": item["id"], "version": version, "model_label": label,
             "model_requested": model, "repetition": repetition}
            for item in QUESTIONS for version in ("A", "B")
            for label, model in config["models"].items()
            for repetition in range(1, config["repeats"] + 1)]
    random.Random(config["seed"]).shuffle(rows)
    for index, row in enumerate(rows, 1):
        row["execution_order"] = index
        row["run_id"] = f"{config['experiment_id']}-{index:03d}-{uuid.uuid4().hex[:8]}"
    assert len(rows) == 2 * 2 * len(config["models"]) * config["repeats"]
    assert len({r["run_id"] for r in rows}) == len(rows)
    return rows


def _sentences(answer):
    return [x.strip() for x in SENTENCES.split(re.sub(r"\s+", " ", answer.strip())) if x.strip()]


def _literature_prompt_records(records):
    return [{"cite": record["cite"], "text": record["text"]} for record in records]


def score_answer(answer, expected, records, gate=None, potentially_truncated=False):
    """Only objective syntax and exact template fields; semantic support stays ungraded."""
    sentences = _sentences(answer)
    ids = ROW_CITE.findall(answer)
    allowed_ids = {expected["target_row_id"], expected["control_row_id"]}
    data_citation_format = not re.search(r"\(DATA:[^)]*\)", answer.replace("\r", "")) or all(
        x in allowed_ids for x in ids)
    data_citation_format = data_citation_format and len(ids) == answer.count("(DATA:")
    first = sentences[0] if sentences else ""
    pattern = (r"^In " + re.escape(expected["comparison"]) + r", " + re.escape(expected["gene"]) +
               r" has log2 fold change ([+-]?\d+(?:\.\d+)?) and adjusted p-value (\d+(?:\.\d+)?), " +
               r"indicating (significant upregulation|significant downregulation|no significant change) " +
               re.escape("(DATA:" + expected["target_row_id"] + ")") + r"\.$")
    match = re.fullmatch(pattern, first)
    fields = {"value": None, "p_value": None, "direction": None}
    if match:
        fields = {"value": float(match[1]) == expected["log2_fold_change"],
                  "p_value": float(match[2]) == expected["adjusted_p_value"],
                  "direction": match[3] == "significant " + expected["direction"]}
    second = sentences[1] if len(sentences) > 1 else ""
    control_pattern = (r"^In the same comparison, " + re.escape(expected["control_gene"]) +
                       r" has log2 fold change ([+-]?\d+(?:\.\d+)?) and adjusted p-value (\d+(?:\.\d+)?), " +
                       r"indicating (no significant change|significant upregulation|significant downregulation) " +
                       re.escape("(DATA:" + expected["control_row_id"] + ")") + r"\.$")
    control_match = re.fullmatch(control_pattern, second)
    control_fields = {"value": None, "p_value": None, "significance": None}
    if control_match:
        control_fields = {"value": float(control_match[1]) == expected["control_log2_fold_change"],
                          "p_value": float(control_match[2]) == expected["control_adjusted_p_value"],
                          "significance": control_match[3] == "no significant change"}
    # An absent or differently worded field is a template failure, not a semantic judgment.
    literature_cites = citations_in(answer)
    valid_endings = [r["cite"] + "." for r in records] + [f"(DATA:{x})." for x in allowed_ids]
    terminal = all(any(sentence.endswith(ending) for ending in valid_endings) for sentence in sentences)
    plain = not any(re.match(r"\s*(#|[-*+]\s|\d+\.\s|\|)", line) for line in answer.splitlines())
    return {"template_parsed": bool(match), "control_template_parsed": bool(control_match),
            "target_value_correct": fields["value"],
            "target_p_correct": fields["p_value"], "target_direction_correct": fields["direction"],
            "control_value_correct": control_fields["value"],
            "control_p_correct": control_fields["p_value"],
            "control_significance_correct": control_fields["significance"],
            "target_row_cited": expected["target_row_id"] in ids,
            "control_row_cited": expected["control_row_id"] in ids,
            "data_citation_format": data_citation_format,
            "literature_citation_count": len(literature_cites),
            "literature_citation_present": len(literature_cites) > 0,
            "literature_gate_pass": gate["pass"] if gate is not None else None,
            "literature_gate_failed": gate["failed"] if gate is not None else None,
            "sentence_count": len(sentences), "potentially_truncated": potentially_truncated,
            "format_gradable": not potentially_truncated,
            "five_sentence_limit": None if potentially_truncated else 1 <= len(sentences) <= 5,
            "plain_prose": None if potentially_truncated else plain,
            "terminal_citation_pattern": None if potentially_truncated else terminal,
            "separation_of_sources": None,
            "literature_semantic_support": None, "unsupported_mechanism": None,
            "unsupported_causality": None, "invented_gene_function": None}


def _payload(model, question, records, system, max_tokens):
    # Keep every frozen chunk's exact text, order, and citation. Retrieval metadata
    # is retained in records.json but is unnecessary for answering and costly here.
    prompt_records = _literature_prompt_records(records)
    user = (f"Question: {question}\n\nChunks retrieval returned, as JSON:\n" +
            json.dumps(prompt_records, ensure_ascii=False, separators=(",", ":")))
    return {"model": model, "temperature": TEMPERATURE, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}


def _request(model, question, records, system, run_dir, max_tokens):
    """Same OpenRouter chat endpoint and retry policy as ask.answer."""
    payload = _payload(model, question, records, system, max_tokens)
    _write_json(run_dir / "request.json", payload)
    attempts = []
    for n in range(1, RETRIES + 1):
        attempt = {"number": n, "timestamp": _now()}
        try:
            response = requests.post(ENDPOINT,
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                timeout=300, json=payload)
            attempt["http_status"] = response.status_code
            try:
                raw_body = response.json()
            except ValueError:
                raw_body = {"raw_text": response.text}
            _write_json(run_dir / f"attempt-{n:02d}-response.json", raw_body)
            message = str(raw_body.get("error", {}).get("message", "")) if isinstance(raw_body, dict) else ""
            if response.status_code == 402 and "in-flight requests" in message.casefold() and n < RETRIES:
                attempt["retry_reason"] = "in_flight_credit_reservation"
                attempts.append(attempt)
                time.sleep(min(20 * n, 60))
                continue
            if response.status_code in TRANSIENT and n < RETRIES:
                attempts.append(attempt)
                time.sleep(2 ** (n - 1))
                continue
            response.raise_for_status()
            body = raw_body
            _write_json(run_dir / "response.json", body)
            if "error" in body:
                raise RuntimeError(f"OpenRouter: {body['error']}")
            attempts.append(attempt)
            return body["choices"][0]["message"]["content"], body, attempts
        except Exception as exc:
            attempt["error"] = repr(exc)
            attempts.append(attempt)
            if n == RETRIES or attempt.get("http_status") not in TRANSIENT:
                raise RequestFailure(repr(exc), attempts) from exc
            time.sleep(2 ** (n - 1))
    raise AssertionError("retry loop exhausted")


class RequestFailure(Exception):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = attempts


def _usage_calibration(root):
    """Observed token/character ratios from saved successful local requests."""
    found = {model: {"input_ratio": [], "output_tokens": []} for model in MODELS.values()}
    base = Path(root) / "reports" / "runs" / "numerical_evidence"
    for path in base.glob("*/results.json"):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for row in rows:
            model = row.get("model_requested")
            usage = row.get("usage") or {}
            if row.get("status") != "success" or model not in found or not usage.get("prompt_tokens"):
                continue
            try:
                payload = json.loads((Path(row["trial_dir"]) / "request.json").read_text(encoding="utf-8"))
            except (OSError, ValueError, KeyError):
                continue
            chars = sum(len(message["content"]) for message in payload["messages"])
            if chars:
                found[model]["input_ratio"].append(usage["prompt_tokens"] / chars)
            if usage.get("completion_tokens") is not None:
                found[model]["output_tokens"].append(usage["completion_tokens"])
    return found


def estimate_usage(config, matrix, evidence, system):
    """Pre-call estimates; cost remains unavailable without verified pricing."""
    calibration = _usage_calibration(config["root"])
    rows = []
    for label, model in config["models"].items():
        plans = [plan for plan in matrix if plan["model_requested"] == model]
        observed = calibration[model]
        ratio = statistics.median(observed["input_ratio"]) if observed["input_ratio"] else 0.28
        output_typical = min(config["max_tokens"], round(statistics.median(observed["output_tokens"]))) if observed["output_tokens"] else min(config["max_tokens"], 250)
        input_estimates = []
        for plan in plans:
            item = next(q for q in QUESTIONS if q["id"] == plan["question_id"])
            payload = _payload(model, _prompt(item, datasets_for(item)[plan["version"]]),
                               evidence[item["id"]], system, config["max_tokens"])
            chars = sum(len(message["content"]) for message in payload["messages"])
            input_estimates.append(round(chars * ratio))
        rows.append({"model_label": label, "model_id": model, "calls": len(plans),
                     "estimated_input_tokens_per_call_min": min(input_estimates),
                     "estimated_input_tokens_per_call_max": max(input_estimates),
                     "estimated_input_tokens_total": sum(input_estimates),
                     "estimated_output_tokens_per_call": output_typical,
                     "estimated_output_tokens_total": output_typical * len(plans),
                     "output_token_cap_per_call": config["max_tokens"],
                     "output_token_cap_total": config["max_tokens"] * len(plans),
                     "estimate_basis": "median prior observed model tokens/character" if observed["input_ratio"] else "fallback 0.28 tokens/character",
                     "cost_estimate_usd": None})
    return pd.DataFrame(rows)


def _review_template(results, run_dir):
    columns = ["run_id", "question_id", "version", "model_label", "repetition", "potentially_truncated", "sentence_number",
               "sentence", "dataset_row_support_yes_no_unclear", "control_numeric_accuracy_yes_no_unclear",
               "literature_claim_yes_no", "literature_support_yes_no_unclear",
               "dataset_literature_separated_yes_no_unclear", "unsupported_mechanism_yes_no",
               "causal_overclaim_yes_no", "invented_function_yes_no", "unchanged_evidence_preserved_yes_no_unclear",
               "reviewer_notes"]
    with (run_dir / "manual_review_template.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in results:
            if row["status"] != "success":
                continue
            for n, sentence in enumerate(_sentences(row["answer"]), 1):
                writer.writerow({"run_id": row["run_id"], "question_id": row["question_id"],
                                 "version": row["version"], "model_label": row["model_label"],
                                 "repetition": row["repetition"],
                                 "potentially_truncated": row.get("potentially_truncated", False),
                                 "sentence_number": n,
                                 "sentence": sentence})
    with (run_dir / "paired_review_template.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fields = ["question_id", "model_label", "repetition", "run_id_A", "run_id_B",
                  "target_direction_changes_yes_no_unclear", "control_conclusion_stable_yes_no_unclear",
                  "literature_conclusions_stable_yes_no_unclear", "unsupported_new_conclusion_yes_no_unclear",
                  "reviewer_notes"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        successful = [r for r in results if r["status"] == "success"]
        for question in QUESTIONS:
            for label in sorted({r["model_label"] for r in successful}):
                for repetition in sorted({r["repetition"] for r in successful}):
                    pair = {r["version"]: r for r in successful if r["question_id"] == question["id"]
                            and r["model_label"] == label and r["repetition"] == repetition}
                    if set(pair) == {"A", "B"}:
                        writer.writerow({"question_id": question["id"], "model_label": label,
                                         "repetition": repetition, "run_id_A": pair["A"]["run_id"],
                                         "run_id_B": pair["B"]["run_id"]})
    (run_dir / "manual_review_rubric.md").write_text("""# Manual review rubric

Review only successful calls. Leave blank until examined against `frozen_evidence/*/records.json`, the synthetic dataset, and the exact answer. Use yes/no/unclear for every applicable judgment. A valid DOI or DATA citation alone does not establish support.

* Dataset support: cited row actually supports each claimed value, comparison, direction and significance. Check the control's values (0.31 adjusted p-value means nonsignificant). Record any wrong sign, missing unit or invented datum in notes.
* Truncation: inspect `potentially_truncated` and the saved finish reason before judging format. Truncated answers are excluded from automatic format-rate denominators and need separate review.
* Literature support: for each literature claim, inspect the cited chunk text and caveat. Mark no if the cited passage does not entail the claim, including unsupported mechanisms.
* Source separation: distinguish synthetic observations from literature explanations; paper citations must not be portrayed as the source of synthetic numbers.
* Causality and function: mark overclaims when synthetic expression is called proof of causality or cell state; mark invented gene functions absent from retrieved chunks.
* Paired interpretation: use `paired_review_template.csv` to compare A and B for the same question, model and repetition. The target direction must reverse; the unchanged control and any literature-supported conclusion should remain stable. Note conflicts and any claim the frozen literature cannot decide.

Do not compute semantic model rankings until this review is complete. Empty fields mean ungraded, not passing.
""", encoding="utf-8")


def _summary(results):
    good = [r for r in results if r["status"] == "success"]
    rows = []
    for label in sorted({r["model_label"] for r in results}):
        subset = [r for r in good if r["model_label"] == label]
        def rate(key):
            vals = [r["checks"].get(key) for r in subset if r["checks"].get(key) is not None]
            return sum(bool(x) for x in vals) / len(vals) if vals else None
        pairs = []
        for q in QUESTIONS:
            for rep in sorted({r["repetition"] for r in subset}):
                pair = {r["version"]: r for r in subset if r["question_id"] == q["id"] and r["repetition"] == rep}
                if set(pair) == {"A", "B"}:
                    pairs.append(all(pair[v]["checks"]["target_direction_correct"] is True for v in ("A", "B")))
        repeat_variants = []
        for q in QUESTIONS:
            for version in ("A", "B"):
                same = [r["answer"] for r in subset if r["question_id"] == q["id"] and r["version"] == version]
                if len(same) >= 2:
                    repeat_variants.append(len(set(same)))
        rows.append({"model_label": label, "successful_calls": len(subset),
                     "failed_calls": sum(r["status"] == "api_failed" for r in results if r["model_label"] == label),
                     "processing_failures": sum(r["status"] == "processing_failed" for r in results if r["model_label"] == label),
                     "model_mismatches": sum(r["status"] == "model_mismatch" for r in results if r["model_label"] == label),
                     "potentially_truncated_calls": sum(bool(r.get("potentially_truncated")) for r in subset),
                     "format_graded_calls": sum(r["checks"].get("format_gradable") is True for r in subset),
                     "data_value_rate": rate("target_value_correct"),
                     "data_p_rate": rate("target_p_correct"),
                     "data_direction_rate": rate("target_direction_correct"),
                     "control_value_rate": rate("control_value_correct"),
                     "control_p_rate": rate("control_p_correct"),
                     "control_significance_rate": rate("control_significance_correct"),
                     "target_row_citation_rate": rate("target_row_cited"),
                     "control_row_citation_rate": rate("control_row_cited"),
                     "literature_citation_gate_rate": rate("literature_gate_pass"),
                     "literature_citation_present_rate": rate("literature_citation_present"),
                     "five_sentence_rate": rate("five_sentence_limit"),
                     "plain_prose_rate": rate("plain_prose"),
                     "paired_direction_rate": sum(pairs) / len(pairs) if pairs else None,
                     "paired_conditions": len(pairs),
                     "mean_distinct_answers_per_repeated_condition": sum(repeat_variants) / len(repeat_variants) if repeat_variants else None,
                     "semantic_support": None, "causal_claim_review": None,
                     "full_interpretation_change": None,
                     "semantic_repeat_consistency": None})
    return pd.DataFrame(rows)


def run_experiment(config, dry_run=True, resume=False):
    """Freeze evidence and execute at most one sequential batch per invocation."""
    if not dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        raise EnvironmentError("Set OPENROUTER_API_KEY before paid model calls")
    validate_design()
    run_dir = Path(config["run_dir"])
    system = CONTRACT + "\n\n" + EXTENSION
    evidence = {}
    hashes = {}
    if resume:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        matrix = json.loads((run_dir / "execution_order.json").read_text(encoding="utf-8"))
        for field in ("experiment_id", "models", "repeats", "k", "corpus", "seed", "max_tokens", "batch_size"):
            if manifest[field] != config[field]:
                raise ValueError(f"Cannot resume with changed {field}")
        if manifest.get("literature_prompt_fields") != ["cite", "text"]:
            raise ValueError("Cannot resume a run made with a different literature prompt format")
        if (manifest.get("question_ids") != [q["id"] for q in QUESTIONS]
                or manifest.get("selected_chunk_ids") != SELECTED_CHUNK_IDS
                or manifest.get("candidate_k") != CANDIDATE_K):
            raise ValueError("Cannot resume a run with a different question or evidence selection")
        if manifest["dry_run"] != dry_run or manifest["system_contract"] != system:
            raise ValueError("Cannot resume with changed dry-run mode or contract")
        for item in QUESTIONS:
            records = json.loads((run_dir / "frozen_evidence" / item["id"] / "records.json").read_text(encoding="utf-8"))
            evidence[item["id"]] = records
            hashes[item["id"]] = _hash(records)
            if hashes[item["id"]] != manifest["retrieval_hashes"][item["id"]]:
                raise ValueError("Frozen evidence hash changed")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        for item in QUESTIONS:
            folder = run_dir / "frozen_evidence" / item["id"]
            folder.mkdir(parents=True)
            candidate_folder = folder / "candidate_retrieval"
            candidate_folder.mkdir()
            if config.get("frozen_from"):
                source = Path(config["frozen_from"]) / "frozen_evidence" / item["id"]
                source_candidates = source / "candidate_retrieval" if (source / "candidate_retrieval" / "records.json").exists() else source
                candidates = json.loads((source_candidates / "records.json").read_text(encoding="utf-8"))
                shutil.copy2(source_candidates / "retrieved.jsonl", candidate_folder / "retrieved.jsonl")
            else:
                candidates = retrieve_logged(candidate_folder, item["retrieval_question"], corpus=config["corpus"], k=CANDIDATE_K)
            if len(candidates) != CANDIDATE_K:
                raise AssertionError("Expected one retrieval of 20 candidate chunks")
            _write_json(candidate_folder / "records.json", candidates)
            selected_ids = set(SELECTED_CHUNK_IDS[item["id"]])
            records = [record for record in candidates if record["chunk_id"] in selected_ids]
            if (len(records) != len(selected_ids) or not 4 <= len(records) <= config["k"]
                    or any(not r.get("cite") or not r.get("text") for r in records)):
                raise AssertionError("Curated evidence is missing or invalid")
            (folder / "retrieved.jsonl").write_text(json.dumps({
                "time": _now(), "query": item["retrieval_question"], "corpus": config["corpus"],
                "k": len(records), "selection_from_k": CANDIDATE_K,
                "chunk_ids": [r["chunk_id"] for r in records]}, ensure_ascii=False) + "\n", encoding="utf-8")
            by_id = {r["chunk_id"]: r for r in records}
            for claim in LITERATURE_CLAIMS[item["id"]]:
                if claim["chunk_id"] not in by_id or claim["excerpt"] not in by_id[claim["chunk_id"]]["text"]:
                    raise AssertionError(f"Claim anchor missing from selected evidence: {claim['claim']}")
            _write_json(folder / "reviewed_literature_claims.json", {
                "review_status": "human reviewed against the selected chunk text; excerpt matching also validated",
                "claims": LITERATURE_CLAIMS[item["id"]],
                "limitations": expected_for(item, "A")["unsupported_conclusions"]})
            evidence[item["id"]] = records
            hashes[item["id"]] = _hash(records)
            _write_json(folder / "records.json", records)
            for version, dataset in datasets_for(item).items():
                _write_json(folder / f"dataset_{version}.json", dataset)
                _write_json(folder / f"expected_{version}.json", expected_for(item, version))
        matrix = _matrix(config)
        manifest = {"experiment_id": config["experiment_id"], "created_utc": _now(),
                    "dry_run": dry_run, "store": str(config["store"]), "models": config["models"],
                    "repeats": config["repeats"], "k": config["k"], "candidate_k": CANDIDATE_K,
                    "selected_chunk_ids": SELECTED_CHUNK_IDS, "question_ids": [q["id"] for q in QUESTIONS],
                    "corpus": config["corpus"],
                    "seed": config["seed"], "temperature": TEMPERATURE, "retries_max": RETRIES,
                    "max_tokens": config["max_tokens"], "batch_size": config["batch_size"],
                    "inter_call_delay_seconds": INTER_CALL_DELAY_SECONDS,
                    "literature_prompt_fields": ["cite", "text"],
                    "frozen_from": config.get("frozen_from"),
                    "system_contract": system, "retrieval_hashes": hashes,
                    "planned_calls": len(matrix), "run_dir": str(run_dir)}
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(run_dir / "execution_order.json", matrix)
        pd.DataFrame(matrix).to_csv(run_dir / "execution_order.csv", index=False)
    # Test exact-field scoring with one correct and one reversed synthetic response.
    for item in QUESTIONS:
        presented = _literature_prompt_records(evidence[item["id"]])
        assert 4 <= len(presented) <= 5
        assert all(presented[i]["cite"] == source["cite"] and presented[i]["text"] == source["text"]
                   for i, source in enumerate(evidence[item["id"]]))
        for version in ("A", "B"):
            exp = expected_for(item, version)
            fixture = (f"In {exp['comparison']}, {exp['gene']} has log2 fold change {exp['log2_fold_change']:+.1f} "
                       f"and adjusted p-value 0.01, indicating significant {exp['direction']} "
                       f"(DATA:{exp['target_row_id']}). "
                       f"In the same comparison, {exp['control_gene']} has log2 fold change "
                       f"{exp['control_log2_fold_change']:+.1f} and adjusted p-value 0.31, "
                       f"indicating no significant change (DATA:{exp['control_row_id']}).")
            check = score_answer(fixture, exp, evidence[item["id"]])
            assert all(check[k] is True for k in ("template_parsed", "target_value_correct", "target_p_correct",
                                                  "target_direction_correct", "target_row_cited",
                                                  "control_template_parsed", "control_value_correct",
                                                  "control_p_correct", "control_significance_correct"))
            wrong = score_answer(fixture.replace(f"{exp['log2_fold_change']:+.1f}", "0.0"), exp, evidence[item["id"]])
            assert wrong["target_value_correct"] is False
    if resume:
        preflight = pd.read_csv(run_dir / "preflight_usage.csv")
    else:
        preflight = estimate_usage(config, matrix, evidence, system)
        preflight.to_csv(run_dir / "preflight_usage.csv", index=False)
        _write_json(run_dir / "preflight_usage.json", preflight.astype(object).where(pd.notna(preflight), None).to_dict("records"))
    print(f"Planned model calls: {len(matrix)}. Estimated usage by model (cost unavailable):")
    display(preflight)
    if dry_run:
        _review_template([], run_dir)
        source_kind = "two reused candidate sets" if config.get("frozen_from") else "two one-time candidate retrievals"
        print(f"Dry run validated {len(matrix)} planned calls, {source_kind}, curated frozen evidence, claims, paired inputs, and scoring fixtures. No model calls. {run_dir}")
        return {"run_dir": run_dir, "manifest": manifest, "matrix": pd.DataFrame(matrix),
                "preflight": preflight, "results": None}
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if resume and results_path.exists() else []
    if list(run_dir.glob("calls/**/inflight.json")):
        raise RuntimeError("An earlier request has an unresolved in-flight marker; inspect it before resuming")
    latest = {row["run_id"]: row for row in results}
    pending = [plan for plan in matrix if latest.get(plan["run_id"], {}).get("status")
               not in ("success", "model_mismatch", "processing_failed")]
    attempted = 0
    for plan in pending[:config["batch_size"]]:
        item = next(q for q in QUESTIONS if q["id"] == plan["question_id"])
        version = plan["version"]
        dataset = datasets_for(item)[version]
        expected = expected_for(item, version)
        attempt_id = uuid.uuid4().hex[:12]
        trial = run_dir / "calls" / plan["run_id"] / attempt_id
        trial.mkdir(parents=True)
        _write_json(trial / "inflight.json", {"started_utc": _now(), "run_id": plan["run_id"]})
        (trial / "retrieved.jsonl").write_bytes((run_dir / "frozen_evidence" / item["id"] / "retrieved.jsonl").read_bytes())
        prompt = _prompt(item, dataset)
        base = {**plan, "timestamp_utc": _now(), "retrieval_hash": hashes[item["id"]],
                "dataset": dataset, "expected": expected, "question": item["question"],
                "prompt": prompt, "system_contract": system,
                "settings": {"temperature": TEMPERATURE, "max_tokens": config["max_tokens"], "endpoint": ENDPOINT},
                "attempt_id": attempt_id,
                "trial_dir": str(trial)}
        start = time.perf_counter()
        try:
            answer, body, attempts = _request(plan["model_requested"], prompt, evidence[item["id"]], system, trial,
                                               config["max_tokens"])
            (trial / "answer.md").write_text(answer, encoding="utf-8")
            gate = gate_run(trial, answer)
            served = body.get("model")
            status = "success" if served == plan["model_requested"] else "model_mismatch"
            finish_reason = body.get("choices", [{}])[0].get("finish_reason")
            completion_tokens = (body.get("usage") or {}).get("completion_tokens")
            potentially_truncated = finish_reason in ("length", "max_tokens") or (
                completion_tokens is not None and completion_tokens >= config["max_tokens"])
            row = {**base, "status": status, "answer": answer, "response_id": body.get("id"),
                   "finish_reason": finish_reason, "potentially_truncated": potentially_truncated,
                   "model_served": body.get("model"), "usage": body.get("usage", {}),
                   "cost_usd": body.get("usage", {}).get("cost"), "attempts": attempts,
                   "retry_count": len(attempts) - 1, "gate": gate,
                   "checks": score_answer(answer, expected, evidence[item["id"]], gate,
                                          potentially_truncated=potentially_truncated), "error": None}
        except RequestFailure as exc:
            row = {**base, "status": "api_failed", "answer": None, "response_id": None,
                   "model_served": None, "usage": None, "cost_usd": None, "attempts": exc.attempts,
                   "retry_count": len(exc.attempts) - 1, "gate": None, "checks": None, "error": str(exc)}
        except Exception as exc:
            row = {**base, "status": "processing_failed", "answer": (trial / "answer.md").read_text(encoding="utf-8") if (trial / "answer.md").exists() else None,
                   "response_id": None, "model_served": None, "usage": None, "cost_usd": None,
                   "attempts": [], "retry_count": None, "gate": None, "checks": None, "error": repr(exc)}
        row["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        _write_json(trial / "result.json", row)
        (trial / "inflight.json").unlink()
        results.append(row)
        _write_json(run_dir / "results.json", results)
        pd.json_normalize(results, sep="_").to_csv(run_dir / "results.csv", index=False)
        print(plan["execution_order"], plan["model_label"], item["id"], version, plan["repetition"], row["status"])
        attempted += 1
        if row["status"] == "api_failed" and any(
                a.get("http_status") in (401, 402, 403) or "ProxyError" in a.get("error", "")
                or "ConnectionError" in a.get("error", "") for a in row["attempts"]):
            print("Stopped batch after an access, credit, or connection failure; inspect the saved attempt before resuming.")
            break
        if attempted < config["batch_size"] and row["status"] == "success":
            time.sleep(INTER_CALL_DELAY_SECONDS)
    latest = {row["run_id"]: row for row in results}
    current = [latest[plan["run_id"]] for plan in matrix if plan["run_id"] in latest]
    _review_template(current, run_dir)
    summary = _summary(current)
    summary.to_csv(run_dir / "summary.csv", index=False)
    _write_json(run_dir / "summary.json", summary.astype(object).where(pd.notna(summary), None).to_dict("records"))
    print(f"Batch attempted {attempted}; {sum(r['status']=='success' for r in current)}/{len(matrix)} completed successfully. {run_dir}")
    display(summary)
    return {"run_dir": run_dir, "manifest": manifest, "matrix": pd.DataFrame(matrix),
            "preflight": preflight, "results": pd.DataFrame(current), "summary": summary}
