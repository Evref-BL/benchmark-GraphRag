#!/usr/bin/env python3
"""Evaluate predictions already stored in a dataset result JSON file."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from evaluator_core import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_RESPONSE_TYPE,
    IssueEvaluation,
    build_expected_index,
    collect_repository_file_keys,
    compute_global_metrics,
    compute_metrics,
    extract_predicted_classes,
    filter_expected_index_by_file_keys,
    load_mined_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute precision/recall/F1 from a dataset result JSON file "
            "without calling an LLM."
        )
    )
    parser.add_argument("dataset_result", help="Dataset JSON containing issues[].response.")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path for the detailed evaluation report.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional local repository root for in_repo_only metrics.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=f"Bootstrap samples for confidence intervals (default: {DEFAULT_BOOTSTRAP_SAMPLES}).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"Bootstrap seed (default: {DEFAULT_BOOTSTRAP_SEED}).",
    )
    return parser.parse_args()


def response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    return json.dumps(response, ensure_ascii=False)


def evaluate_issue(
    issue: Dict[str, Any],
    *,
    in_repo_file_keys: Optional[set[str]],
) -> IssueEvaluation:
    expected = build_expected_index(issue)
    predicted = extract_predicted_classes(
        response_text(issue.get("response", "")), DEFAULT_RESPONSE_TYPE
    )
    tp, fp, fn, precision, recall, f1, _ = compute_metrics(expected, predicted)

    evaluation = IssueEvaluation(
        issue_number=int(issue.get("number", issue.get("id", 0))),
        issue_title=str(issue.get("title", "") or ""),
        issue_url=str(issue.get("url", "") or ""),
        expected_java_files=sorted(target.file_path for target in expected.targets.values()),
        predicted_classes=[prediction.raw for prediction in predicted],
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        query="",
        prompt_exact_passed_to_model="",
        status="ok",
        error=None,
        llm_content=response_text(issue.get("response", "")),
    )

    if in_repo_file_keys is not None:
        in_repo_expected = filter_expected_index_by_file_keys(expected, in_repo_file_keys)
        in_tp, in_fp, in_fn, in_precision, in_recall, in_f1, _ = compute_metrics(
            in_repo_expected, predicted
        )
        retained = sorted(target.file_path for target in in_repo_expected.targets.values())
        evaluation.expected_java_files_in_repo = retained
        evaluation.true_positives_in_repo = in_tp
        evaluation.false_positives_in_repo = in_fp
        evaluation.false_negatives_in_repo = in_fn
        evaluation.precision_in_repo = in_precision
        evaluation.recall_in_repo = in_recall
        evaluation.f1_in_repo = in_f1

    return evaluation


def report_issue(issue_eval: IssueEvaluation) -> Dict[str, Any]:
    return {
        "issue_number": issue_eval.issue_number,
        "issue_title": issue_eval.issue_title,
        "issue_url": issue_eval.issue_url,
        "expected_java_files": issue_eval.expected_java_files,
        "predicted_classes": issue_eval.predicted_classes,
        "true_positives": issue_eval.true_positives,
        "false_positives": issue_eval.false_positives,
        "false_negatives": issue_eval.false_negatives,
        "precision": issue_eval.precision,
        "recall": issue_eval.recall,
        "f1": issue_eval.f1,
        "status": issue_eval.status,
        "error": issue_eval.error,
        "llm_content": issue_eval.llm_content,
        "in_repo_only": (
            {
                "expected_java_files": issue_eval.expected_java_files_in_repo,
                "true_positives": issue_eval.true_positives_in_repo,
                "false_positives": issue_eval.false_positives_in_repo,
                "false_negatives": issue_eval.false_negatives_in_repo,
                "precision": issue_eval.precision_in_repo,
                "recall": issue_eval.recall_in_repo,
                "f1": issue_eval.f1_in_repo,
            }
            if issue_eval.expected_java_files_in_repo is not None
            else None
        ),
    }


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset_result).expanduser().resolve()
    try:
        dataset = load_mined_json(dataset_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[ERROR] Cannot read dataset result: {error}", file=sys.stderr)
        return 1

    issues = dataset["issues"]
    missing_response = [
        issue.get("number", "?")
        for issue in issues
        if not isinstance(issue, dict) or "response" not in issue
    ]
    if missing_response:
        print(
            f"[ERROR] {len(missing_response)} issue(s) have no 'response' field.",
            file=sys.stderr,
        )
        return 1

    in_repo_file_keys = None
    project_root = None
    if args.project_root:
        project_root = Path(args.project_root).expanduser().resolve()
        if not project_root.is_dir():
            print(f"[ERROR] Project root not found: {project_root}", file=sys.stderr)
            return 1
        in_repo_file_keys = collect_repository_file_keys(project_root, ".java")

    results = [
        evaluate_issue(issue, in_repo_file_keys=in_repo_file_keys)
        for issue in issues
        if isinstance(issue, dict)
    ]
    summary = compute_global_metrics(
        results,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    report = {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "mode": "dataset_result",
        "benchmark_source_file": str(dataset_path),
        "settings": {
            "response_type": DEFAULT_RESPONSE_TYPE,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "project_root": str(project_root) if project_root else None,
        },
        "summary": summary,
        "issues": [report_issue(result) for result in results],
    }

    output_path = (
        Path(args.output_file).expanduser().resolve()
        if args.output_file
        else dataset_path.with_name(f"{dataset_path.stem}__eval.json")
    )
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as error:
        print(f"[ERROR] Cannot write report: {error}", file=sys.stderr)
        return 1

    micro = summary["micro"]
    print(f"Evaluated issues: {summary['issues_evaluated']}")
    print(
        f"Micro: precision={micro['precision']:.4f}, "
        f"recall={micro['recall']:.4f}, f1={micro['f1']:.4f}"
    )
    print(f"Report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
