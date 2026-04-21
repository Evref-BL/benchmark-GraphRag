#!/usr/bin/env python3
"""Evaluate benchmark issues with GraphRAG (specialization of BaseBenchmarkEvaluator)."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

from evaluator_core import (
    BaseBenchmarkEvaluator,
    DEFAULT_EVAL_OUTPUT_DIRNAME,
    DEFAULT_ISSUE_PROMPT,
    DEFAULT_RESPONSE_TYPE,
    DEFAULT_TIMEOUT_SECONDS,
    REQUIRED_PRE_PROMPT,
    PredictionResult,
    build_expected_index,
    build_query_text,
    extract_predicted_classes,
    load_mined_json,
)

ALLOWED_METHODS = ("local", "drift", "global")


def parse_args() -> argparse.Namespace:
    def parse_method(value: str) -> str:
        method = (value or "").strip().lower()
        if method not in ALLOWED_METHODS:
            allowed = ", ".join(ALLOWED_METHODS)
            raise argparse.ArgumentTypeError(
                f"Invalid method '{value}'. Allowed values: {allowed}."
            )
        return method

    project_root = Path(__file__).resolve().parent.parent
    default_graphrag_dir = project_root / "graphrag"
    default_output_dir = project_root / DEFAULT_EVAL_OUTPUT_DIRNAME
    parser = argparse.ArgumentParser(
        description=(
            "Execute GraphRAG query for each issue in a mined benchmark file and "
            "compute precision/recall/F1 on impacted Java files."
        )
    )
    parser.add_argument("mined_file", help="Path to a mining JSON benchmark file.")
    parser.add_argument(
        "--graphrag-dir",
        default=str(default_graphrag_dir),
        help="Path to the GraphRAG project directory (default: project-root/graphrag).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help=(
            "Directory for evaluation reports "
            f"(default: project-root/{DEFAULT_EVAL_OUTPUT_DIRNAME})."
        ),
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional full output JSON report path (overrides --output-dir).",
    )
    parser.add_argument(
        "--issue-limit",
        type=int,
        default=None,
        help="Optional cap on number of issues evaluated.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout in seconds for each GraphRAG query (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--extra-prompt",
        default=DEFAULT_ISSUE_PROMPT,
        help="Prompt suffix appended after title+description.",
    )
    parser.add_argument(
        "--method",
        type=parse_method,
        default="local",
        help="GraphRAG query method: local, drift, or global (default: local).",
    )
    parser.add_argument(
        "--data-dir",
        default="./output",
        help="GraphRAG data directory passed to --data (default: ./output).",
    )
    parser.add_argument(
        "--include-empty-java",
        action="store_true",
        help="Evaluate issues even when no Java files are expected (default: skip).",
    )
    parser.add_argument(
        "--keep-raw-response",
        action="store_true",
        help="Store full GraphRAG raw response in output JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare prompts/metrics inputs but do not call GraphRAG.",
    )
    return parser.parse_args()


class GraphRAGEvaluator(BaseBenchmarkEvaluator):
    """GraphRAG specialization of the shared evaluator flow."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.graphrag_dir = Path(args.graphrag_dir).expanduser().resolve()
        self.method = args.method
        self.data_dir = args.data_dir
        self.timeout_seconds = args.timeout_seconds

    def evaluator_label(self) -> str:
        return "graphrag"

    def validate_runtime(self) -> None:
        if self.graphrag_dir.is_dir():
            return
        if self.dry_run:
            print(
                f"[WARN] GraphRAG directory not found (dry-run mode): {self.graphrag_dir}",
                file=sys.stderr,
            )
            return
        raise RuntimeError(f"GraphRAG directory not found: {self.graphrag_dir}")

    def predict_for_issue(self, issue: Dict[str, Any], query: str) -> PredictionResult:
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "graphrag",
            "query",
            query,
            "--root",
            ".",
            "--method",
            self.method,
            "--data",
            self.data_dir,
            "--response-type",
            DEFAULT_RESPONSE_TYPE,
        ]
        completed = subprocess.run(
            command,
            cwd=str(self.graphrag_dir),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        output = completed.stdout.strip()
        if completed.returncode != 0:
            stderr_text = completed.stderr.strip()
            message = stderr_text or output or f"GraphRAG query failed with code {completed.returncode}."
            raise RuntimeError(message)
        predicted = extract_predicted_classes(output, DEFAULT_RESPONSE_TYPE)
        return PredictionResult(
            predicted_objects=predicted,
            raw_response=output,
            command=command,
        )

    def default_report_path(self, mined_path: Path, output_dir: Path) -> Path:
        timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return output_dir / f"{mined_path.stem}__graphrag_eval__{timestamp}.json"

    def settings(self) -> Dict[str, Any]:
        return {
            "graphrag_dir": str(self.graphrag_dir),
            "method": self.method,
            "data_dir": self.data_dir,
            "response_type": DEFAULT_RESPONSE_TYPE,
            "required_pre_prompt": REQUIRED_PRE_PROMPT,
            "extra_prompt": self.extra_prompt,
            "timeout_seconds": self.timeout_seconds,
            "issue_limit": self.issue_limit,
            "output_dir": str(self.output_dir),
            "dry_run": self.dry_run,
            "include_empty_java": self.include_empty_java,
            "keep_raw_response": self.keep_raw_response,
        }

    def issue_extra_fields(self, issue_eval) -> Dict[str, Any]:
        return {"prompt_exact_passed_to_graphrag_llm": issue_eval.prompt_exact_passed_to_model}


def main() -> int:
    args = parse_args()
    evaluator = GraphRAGEvaluator(args)
    return evaluator.run()


if __name__ == "__main__":
    raise SystemExit(main())
