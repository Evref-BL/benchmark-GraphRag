#!/usr/bin/env python3
"""Evaluate benchmark issues with colgrep semantic code search."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

from evaluator_core import (
    BaseBenchmarkEvaluator,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EVAL_OUTPUT_DIRNAME,
    DEFAULT_ISSUE_PROMPT,
    DEFAULT_RESPONSE_TYPE,
    DEFAULT_TIMEOUT_SECONDS,
    PredictionResult,
    collect_repository_file_keys,
    extract_predicted_classes,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_output_dir = project_root / DEFAULT_EVAL_OUTPUT_DIRNAME
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate benchmark issues by querying colgrep semantic search and "
            "computing precision/recall/F1 on impacted Java files."
        )
    )
    parser.add_argument("mined_file", help="Path to a mining JSON benchmark file.")
    parser.add_argument(
        "--project-root",
        required=True,
        help="Path to the source project where colgrep will run.",
    )
    parser.add_argument(
        "--colgrep-bin",
        default="colgrep",
        help="colgrep executable (default: colgrep).",
    )
    parser.add_argument(
        "--results",
        type=int,
        default=15,
        help="Top-K results requested from colgrep (default: 15).",
    )
    parser.add_argument(
        "--include-pattern",
        default="*.java",
        help="File include pattern passed to colgrep (default: *.java).",
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
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=f"Bootstrap sample count for 95%% confidence intervals (default: {DEFAULT_BOOTSTRAP_SAMPLES}).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"Bootstrap RNG seed for 95%% confidence intervals (default: {DEFAULT_BOOTSTRAP_SEED}).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout in seconds for each colgrep call (default: {DEFAULT_TIMEOUT_SECONDS}).",
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
        help="Store full raw colgrep JSON response in output report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare prompts/metrics inputs but do not call colgrep.",
    )
    return parser.parse_args()


class ColgrepEvaluator(BaseBenchmarkEvaluator):
    """colgrep specialization of the shared evaluator flow."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.project_root = Path(args.project_root).expanduser().resolve()
        self.colgrep_bin = args.colgrep_bin
        self.results = args.results
        self.include_pattern = args.include_pattern
        self.timeout_seconds = args.timeout_seconds
        self._resolved_colgrep_bin: Optional[str] = None
        self._repo_java_file_keys: Optional[set[str]] = None

    def evaluator_label(self) -> str:
        return "colgrep"

    def validate_runtime(self) -> None:
        if not self.project_root.is_dir():
            raise RuntimeError(f"Project root not found: {self.project_root}")

        resolved = shutil.which(self.colgrep_bin)
        if not resolved:
            raise RuntimeError(
                f"Unable to find colgrep executable '{self.colgrep_bin}' in PATH."
            )
        self._resolved_colgrep_bin = resolved
        self._repo_java_file_keys = collect_repository_file_keys(self.project_root, ".java")

    def _normalize_unit_file(self, raw_file: str) -> str:
        file_path = Path(raw_file).expanduser()
        try:
            if not file_path.is_absolute():
                file_path = (self.project_root / file_path).resolve()
            else:
                file_path = file_path.resolve()
        except Exception:  # noqa: BLE001
            return raw_file.replace("\\", "/")

        try:
            rel = file_path.relative_to(self.project_root)
            return rel.as_posix()
        except ValueError:
            return file_path.as_posix()

    def _extract_paths_from_colgrep_json(self, payload: Any) -> List[str]:
        if not isinstance(payload, list):
            return []
        paths: List[str] = []
        seen: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                continue
            unit = item.get("unit")
            if not isinstance(unit, dict):
                continue
            raw_file = unit.get("file")
            if not isinstance(raw_file, str) or not raw_file.strip():
                continue
            normalized = self._normalize_unit_file(raw_file.strip())
            if normalized in seen:
                continue
            seen.add(normalized)
            paths.append(normalized)
        return paths

    def predict_for_issue(self, issue: Dict[str, Any], query: str) -> PredictionResult:
        command = [
            self._resolved_colgrep_bin or self.colgrep_bin,
            "--json",
            "--results",
            str(self.results),
            "--include",
            self.include_pattern,
            query,
        ]
        completed = subprocess.run(
            command,
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        stdout_text = completed.stdout.strip()
        if completed.returncode != 0:
            stderr_text = completed.stderr.strip()
            message = stderr_text or stdout_text or f"colgrep failed with code {completed.returncode}."
            raise RuntimeError(message)

        try:
            response_json = json.loads(stdout_text) if stdout_text else []
        except json.JSONDecodeError as error:
            raise RuntimeError(f"colgrep did not return valid JSON: {stdout_text[:1000]}") from error

        candidate_paths = self._extract_paths_from_colgrep_json(response_json)
        synthetic_response = {"classes": [{"path": path} for path in candidate_paths]}
        synthetic_response_text = json.dumps(synthetic_response, ensure_ascii=False)
        predicted = extract_predicted_classes(synthetic_response_text, DEFAULT_RESPONSE_TYPE)

        return PredictionResult(
            predicted_objects=predicted,
            raw_response=stdout_text,
            command=command,
            llm_content=synthetic_response_text,
            request_payload={
                "query": query,
                "results": self.results,
                "include_pattern": self.include_pattern,
                "candidate_paths_count": len(candidate_paths),
            },
        )

    def default_report_path(self, mined_path: Path, output_dir: Path) -> Path:
        timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return output_dir / f"{mined_path.stem}__colgrep_eval__{timestamp}.json"

    def settings(self) -> Dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "colgrep_bin": self.colgrep_bin,
            "resolved_colgrep_bin": self._resolved_colgrep_bin,
            "results": self.results,
            "include_pattern": self.include_pattern,
            "response_type": DEFAULT_RESPONSE_TYPE,
            "extra_prompt": self.extra_prompt,
            "timeout_seconds": self.timeout_seconds,
            "issue_limit": self.issue_limit,
            "output_dir": str(self.output_dir),
            "dry_run": self.dry_run,
            "include_empty_java": self.include_empty_java,
            "keep_raw_response": self.keep_raw_response,
        }

    def in_repo_reference_keys(self) -> Optional[set[str]]:
        return self._repo_java_file_keys

    def issue_extra_fields(self, issue_eval) -> Dict[str, Any]:
        return {"prompt_exact_passed_to_colgrep": issue_eval.prompt_exact_passed_to_model}


def main() -> int:
    args = parse_args()
    evaluator = ColgrepEvaluator(args)
    return evaluator.run()


if __name__ == "__main__":
    raise SystemExit(main())
