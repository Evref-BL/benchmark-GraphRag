#!/usr/bin/env python3
"""Evaluate benchmark issues with GraphRAG (specialization of BaseBenchmarkEvaluator)."""

from __future__ import annotations

import argparse
import datetime as dt
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
    REQUIRED_PRE_PROMPT,
    PredictionResult,
    collect_repository_file_keys,
    extract_predicted_classes,
)

ALLOWED_METHODS = ("local", "drift", "global", "basic")
ALLOWED_EXECUTION_MODES = ("auto", "venv", "uv")


def parse_args() -> argparse.Namespace:
    def parse_method(value: str) -> str:
        method = (value or "").strip().lower()
        if method not in ALLOWED_METHODS:
            allowed = ", ".join(ALLOWED_METHODS)
            raise argparse.ArgumentTypeError(
                f"Invalid method '{value}'. Allowed values: {allowed}."
            )
        return method

    def parse_execution_mode(value: str) -> str:
        mode = (value or "").strip().lower()
        if mode not in ALLOWED_EXECUTION_MODES:
            allowed = ", ".join(ALLOWED_EXECUTION_MODES)
            raise argparse.ArgumentTypeError(
                f"Invalid execution mode '{value}'. Allowed values: {allowed}."
            )
        return mode

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
        help="GraphRAG query method: local, drift, global, or basic (default: local).",
    )
    parser.add_argument(
        "--execution-mode",
        type=parse_execution_mode,
        default="auto",
        help=(
            "How to invoke GraphRAG: auto, venv, or uv (default: auto). "
            "Auto prefers <graphrag-dir>/.venv/bin/graphrag, then falls back to uv."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="./output",
        help="GraphRAG data directory passed to --data (default: ./output).",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help=(
            "Optional source project root used only for in_repo_only metrics "
            "(expected files intersected with files present in this project)."
        ),
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
        self.execution_mode = args.execution_mode
        self.data_dir = args.data_dir
        self.timeout_seconds = args.timeout_seconds
        self.project_root = (
            Path(args.project_root).expanduser().resolve() if getattr(args, "project_root", None) else None
        )
        self._repo_java_file_keys: Optional[set[str]] = None
        self._resolved_command_prefix: Optional[List[str]] = None
        self._resolved_execution_mode: Optional[str] = None

    def evaluator_label(self) -> str:
        return "graphrag"

    def _resolve_command_prefix(self) -> List[str]:
        if self._resolved_command_prefix is not None:
            return self._resolved_command_prefix

        venv_graphrag = self.graphrag_dir / ".venv" / "bin" / "graphrag"
        if self.execution_mode in ("auto", "venv"):
            if venv_graphrag.is_file():
                self._resolved_execution_mode = "venv"
                self._resolved_command_prefix = [str(venv_graphrag)]
                return self._resolved_command_prefix
            if self.execution_mode == "venv":
                raise RuntimeError(f"GraphRAG venv binary not found: {venv_graphrag}")

        if self.execution_mode in ("auto", "uv"):
            uv_bin = shutil.which("uv")
            if uv_bin:
                self._resolved_execution_mode = "uv"
                self._resolved_command_prefix = [uv_bin, "run", "python", "-m", "graphrag"]
                return self._resolved_command_prefix
            if self.execution_mode == "uv":
                raise RuntimeError("uv executable not found in PATH.")

        raise RuntimeError(
            "Unable to resolve GraphRAG executable. "
            "Use --execution-mode venv with a valid .venv/bin/graphrag or install uv."
        )

    def validate_runtime(self) -> None:
        graphrag_dir_ready = True
        if not self.graphrag_dir.is_dir():
            if self.dry_run:
                print(
                    f"[WARN] GraphRAG directory not found (dry-run mode): {self.graphrag_dir}",
                    file=sys.stderr,
                )
                graphrag_dir_ready = False
            else:
                raise RuntimeError(f"GraphRAG directory not found: {self.graphrag_dir}")

        if graphrag_dir_ready:
            try:
                self._resolve_command_prefix()
            except RuntimeError as error:
                if self.dry_run:
                    print(f"[WARN] {error}", file=sys.stderr)
                else:
                    raise

        if self.project_root is not None:
            if not self.project_root.is_dir():
                if self.dry_run:
                    print(
                        f"[WARN] Project root not found for in_repo_only metrics (dry-run mode): "
                        f"{self.project_root}",
                        file=sys.stderr,
                    )
                else:
                    raise RuntimeError(
                        f"Project root not found for in_repo_only metrics: {self.project_root}"
                    )
            else:
                self._repo_java_file_keys = collect_repository_file_keys(self.project_root, ".java")

    def predict_for_issue(self, issue: Dict[str, Any], query: str) -> PredictionResult:
        command = [
            *self._resolve_command_prefix(),
            "query",
            "--root",
            ".",
            "--method",
            self.method,
            "--data",
            self.data_dir,
            "--response-type",
            DEFAULT_RESPONSE_TYPE,
            query,
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
            "execution_mode": self.execution_mode,
            "resolved_execution_mode": self._resolved_execution_mode,
            "resolved_command_prefix": self._resolved_command_prefix,
            "data_dir": self.data_dir,
            "project_root": (str(self.project_root) if self.project_root else None),
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

    def in_repo_reference_keys(self) -> Optional[set[str]]:
        return self._repo_java_file_keys

    def issue_extra_fields(self, issue_eval) -> Dict[str, Any]:
        return {"prompt_exact_passed_to_graphrag_llm": issue_eval.prompt_exact_passed_to_model}


def main() -> int:
    args = parse_args()
    evaluator = GraphRAGEvaluator(args)
    return evaluator.run()


if __name__ == "__main__":
    raise SystemExit(main())
