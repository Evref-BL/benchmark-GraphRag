#!/usr/bin/env python3
"""Semi-automatic evaluator: user pastes model output issue by issue."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys
from typing import Any, Dict, List

from evaluator_core import (
    BaseBenchmarkEvaluator,
    DEFAULT_EVAL_OUTPUT_DIRNAME,
    DEFAULT_ISSUE_PROMPT,
    DEFAULT_RESPONSE_TYPE,
    EvaluationStopped,
    PredictionResult,
    extract_predicted_classes,
)

DEFAULT_END_TOKEN = "EOF"
STOP_TOKEN = "STOP"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_output_dir = project_root / DEFAULT_EVAL_OUTPUT_DIRNAME
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a benchmark in semi-automatic mode: the script builds each query, "
            "shows it, then waits for manual pasted model output."
        )
    )
    parser.add_argument("mined_file", help="Path to a mining JSON benchmark file.")
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
        "--extra-prompt",
        default=DEFAULT_ISSUE_PROMPT,
        help="Prompt suffix appended after title+description.",
    )
    parser.add_argument(
        "--include-empty-java",
        action="store_true",
        help="Evaluate issues even when no Java files are expected (default: skip).",
    )
    parser.add_argument(
        "--keep-raw-response",
        action="store_true",
        help="Store full pasted manual response in output JSON.",
    )
    parser.add_argument(
        "--end-token",
        default=DEFAULT_END_TOKEN,
        help=(
            "Optional line token to end pasted response "
            f"(default: {DEFAULT_END_TOKEN}). A blank line also ends input. "
            f"Type {STOP_TOKEN} to stop the full evaluation early."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts but skip manual input and prediction parsing.",
    )
    return parser.parse_args()


class ManualInputEvaluator(BaseBenchmarkEvaluator):
    """Interactive evaluator: manual response entry per issue."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.end_token = args.end_token

    def evaluator_label(self) -> str:
        return "manual_input"

    def validate_runtime(self) -> None:
        return None

    def _read_multiline_response(self) -> str:
        lines: List[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip().upper() == STOP_TOKEN:
                raise EvaluationStopped("Stopped by user request.")
            if line.strip() == "":
                break
            if self.end_token.strip() and line.strip() == self.end_token:
                break
            lines.append(line)
        return "\n".join(lines).strip()

    def predict_for_issue(self, issue: Dict[str, Any], query: str) -> PredictionResult:
        issue_number = issue.get("number")
        issue_title = str(issue.get("title", "") or "")

        print("\n" + "=" * 100)
        print(f"Issue #{issue_number}: {issue_title}")
        print("-" * 100)
        print("Query to send to the model:\n")
        print(query)
        print("-" * 100)
        print(
            (
                "Paste the model output below, then press Enter on an empty line to continue. "
                f"(Optional: type '{self.end_token}' on a single line.) "
                f"Type '{STOP_TOKEN}' to stop now and save partial results."
            ),
            flush=True,
        )
        manual_response = self._read_multiline_response()
        if not manual_response:
            print(
                "[WARN] Empty manual response received; this issue will have zero predicted classes.",
                file=sys.stderr,
            )
        predicted = extract_predicted_classes(manual_response, DEFAULT_RESPONSE_TYPE)
        return PredictionResult(
            predicted_objects=predicted,
            raw_response=manual_response,
            llm_content=manual_response,
        )

    def default_report_path(self, mined_path: Path, output_dir: Path) -> Path:
        timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return output_dir / f"{mined_path.stem}__manual_eval__{timestamp}.json"

    def settings(self) -> Dict[str, Any]:
        return {
            "mode": "manual_input",
            "response_type": DEFAULT_RESPONSE_TYPE,
            "extra_prompt": self.extra_prompt,
            "issue_limit": self.issue_limit,
            "output_dir": str(self.output_dir),
            "include_empty_java": self.include_empty_java,
            "keep_raw_response": self.keep_raw_response,
            "end_token": self.end_token,
            "dry_run": self.dry_run,
        }

    def run(self) -> int:
        code = super().run()
        if code == 0 and self.last_output_path and self.last_report:
            summary = self.last_report.get("summary", {})
            micro = summary.get("micro", {})
            print("\n[DONE] Manual evaluation completed.")
            print(f"[DONE] Results file: {self.last_output_path}")
            print(
                "[DONE] Final micro metrics -> "
                f"precision: {micro.get('precision', 0.0):.4f}, "
                f"recall: {micro.get('recall', 0.0):.4f}, "
                f"f1: {micro.get('f1', 0.0):.4f}"
            )
        return code


def main() -> int:
    args = parse_args()
    evaluator = ManualInputEvaluator(args)
    return evaluator.run()


if __name__ == "__main__":
    raise SystemExit(main())
