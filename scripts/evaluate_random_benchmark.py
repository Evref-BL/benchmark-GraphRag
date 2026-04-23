#!/usr/bin/env python3
"""Evaluate benchmark issues with a random file-path baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional

from evaluator_core import (
    BaseBenchmarkEvaluator,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EVAL_OUTPUT_DIRNAME,
    DEFAULT_ISSUE_PROMPT,
    DEFAULT_RESPONSE_TYPE,
    PredictedClass,
    PredictionResult,
    build_expected_index,
    java_path_to_fqcn_candidates,
    normalize_path,
)

DEFAULT_FILE_EXTENSION = ".java"
DEFAULT_SAMPLING_STRATEGY = "size-matched"
SAMPLING_STRATEGIES = ("uniform", "size-matched")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_output_dir = project_root / DEFAULT_EVAL_OUTPUT_DIRNAME
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate benchmark issues with a random baseline: for each issue, "
            "draw a random N and sample N file paths from the target project."
        )
    )
    parser.add_argument("mined_file", help="Path to a mining JSON benchmark file.")
    parser.add_argument(
        "--project-root",
        required=True,
        help="Path of the source project used to collect candidate file paths.",
    )
    parser.add_argument(
        "--file-extension",
        default=DEFAULT_FILE_EXTENSION,
        help=f"Candidate file extension to sample from (default: {DEFAULT_FILE_EXTENSION}).",
    )
    parser.add_argument(
        "--random-n-min",
        type=int,
        default=1,
        help="Minimum random N sampled per issue (default: 1).",
    )
    parser.add_argument(
        "--random-n-max",
        type=int,
        default=None,
        help="Maximum random N sampled per issue (default: pool size).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible runs.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=SAMPLING_STRATEGIES,
        default=DEFAULT_SAMPLING_STRATEGY,
        help=(
            "How random N is selected per issue: "
            "'uniform' keeps random-n-min/max behavior, "
            "'size-matched' sets N to the issue expected target count found in the project pool "
            f"(default: {DEFAULT_SAMPLING_STRATEGY})."
        ),
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
        help="Store generated random response JSON in output report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts/metrics inputs but skip random prediction generation.",
    )

    args = parser.parse_args()
    if args.random_n_min < 0:
        parser.error("--random-n-min must be >= 0.")
    if args.random_n_max is not None and args.random_n_max < 0:
        parser.error("--random-n-max must be >= 0 when provided.")
    if args.random_n_max is not None and args.random_n_max < args.random_n_min:
        parser.error("--random-n-max must be >= --random-n-min.")
    return args


def normalize_extension(raw_extension: str) -> str:
    ext = (raw_extension or "").strip().lower()
    if not ext:
        raise ValueError("file extension cannot be empty.")
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def collect_project_file_pool(project_root: Path, extension: str) -> List[str]:
    pool: List[str] = []
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() != extension:
            continue
        rel_path = path.relative_to(project_root).as_posix()
        pool.append(rel_path)
    return sorted(set(pool))


class RandomBenchmarkEvaluator(BaseBenchmarkEvaluator):
    """Random baseline specialization of the shared evaluator flow."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.project_root = Path(args.project_root).expanduser().resolve()
        self.file_extension = normalize_extension(args.file_extension)
        self.random_n_min = args.random_n_min
        self.random_n_max = args.random_n_max
        self.seed = args.seed
        self.sampling_strategy = args.sampling_strategy
        self.rng = random.Random(self.seed)
        self.file_pool: List[str] = []
        self.file_pool_keys: set[str] = set()

    def evaluator_label(self) -> str:
        return "random_baseline"

    def validate_runtime(self) -> None:
        if not self.project_root.is_dir():
            raise RuntimeError(f"Project root not found: {self.project_root}")

        self.file_pool = collect_project_file_pool(self.project_root, self.file_extension)
        self.file_pool_keys = {normalize_path(path) for path in self.file_pool}
        if not self.file_pool:
            raise RuntimeError(
                f"No files found in project root {self.project_root} with extension {self.file_extension}."
            )

        if self.random_n_min > len(self.file_pool):
            raise RuntimeError(
                "random-n-min is greater than available file pool size "
                f"({self.random_n_min} > {len(self.file_pool)})."
            )

        print(
            "[INFO] Random evaluator pool prepared: "
            f"{len(self.file_pool)} files matching *{self.file_extension} under {self.project_root}",
            file=sys.stderr,
        )

    def _sample_size_uniform(self) -> int:
        pool_size = len(self.file_pool)
        upper = self.random_n_max if self.random_n_max is not None else pool_size
        upper = min(upper, pool_size)
        lower = min(self.random_n_min, pool_size)
        if upper < lower:
            upper = lower
        return self.rng.randint(lower, upper) if upper > 0 else 0

    def _sample_size_size_matched(self, issue: Dict[str, Any]) -> int:
        """Use issue cardinality as N after intersecting expected files with the pool."""
        # Size-matched baseline: same expected cardinality, but only on files that
        # can actually be sampled from this evaluator's local project pool.
        expected_index = build_expected_index(issue)
        expected_in_pool = [
            file_key for file_key in expected_index.targets.keys() if file_key in self.file_pool_keys
        ]
        return min(len(expected_in_pool), len(self.file_pool))

    def _sample_size(self, issue: Dict[str, Any]) -> int:
        if self.sampling_strategy == "size-matched":
            return self._sample_size_size_matched(issue)
        return self._sample_size_uniform()

    def _build_predicted_objects(self, sampled_paths: List[str]) -> List[PredictedClass]:
        seen: set[str] = set()
        predictions: List[PredictedClass] = []
        for rel_path in sampled_paths:
            path_key = normalize_path(rel_path)
            if not path_key:
                continue
            prediction_id = f"path:{path_key}"
            if prediction_id in seen:
                continue
            seen.add(prediction_id)
            simple_key = Path(rel_path).stem.lower() or None
            fqcn_key: Optional[str] = None
            if rel_path.lower().endswith(".java"):
                fqcn_candidates = sorted(java_path_to_fqcn_candidates(rel_path))
                if fqcn_candidates:
                    fqcn_key = fqcn_candidates[0]
            predictions.append(
                PredictedClass(
                    prediction_id=prediction_id,
                    raw=rel_path,
                    path_key=path_key,
                    simple_key=simple_key,
                    fqcn_key=fqcn_key,
                )
            )
        return predictions

    def predict_for_issue(self, issue: Dict[str, Any], query: str) -> PredictionResult:
        random_n = self._sample_size(issue)
        sampled_paths = self.rng.sample(self.file_pool, random_n) if random_n > 0 else []
        response_payload = {"classes": [{"path": path} for path in sampled_paths]}
        response_text = json.dumps(response_payload, ensure_ascii=False)
        predicted_objects = self._build_predicted_objects(sampled_paths)

        return PredictionResult(
            predicted_objects=predicted_objects,
            raw_response=response_text,
            llm_content=response_text,
            command=[
                "random_baseline",
                f"--project-root={self.project_root}",
                f"--file-extension={self.file_extension}",
                f"--sampling-strategy={self.sampling_strategy}",
                f"--random-n={random_n}",
            ],
            request_payload={
                "query": query,
                "project_root": str(self.project_root),
                "file_extension": self.file_extension,
                "sampling_strategy": self.sampling_strategy,
                "pool_size": len(self.file_pool),
                "random_n": random_n,
                "seed": self.seed,
            },
        )

    def default_report_path(self, mined_path: Path, output_dir: Path) -> Path:
        timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return output_dir / f"{mined_path.stem}__random_eval__{timestamp}.json"

    def settings(self) -> Dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "file_extension": self.file_extension,
            "file_pool_size": len(self.file_pool),
            "random_n_min": self.random_n_min,
            "random_n_max": self.random_n_max,
            "seed": self.seed,
            "sampling_strategy": self.sampling_strategy,
            "response_type": DEFAULT_RESPONSE_TYPE,
            "extra_prompt": self.extra_prompt,
            "issue_limit": self.issue_limit,
            "output_dir": str(self.output_dir),
            "dry_run": self.dry_run,
            "include_empty_java": self.include_empty_java,
            "keep_raw_response": self.keep_raw_response,
        }

    def in_repo_reference_keys(self) -> Optional[set[str]]:
        return self.file_pool_keys

    def issue_extra_fields(self, issue_eval) -> Dict[str, Any]:
        return {"prompt_exact_passed_to_random_evaluator": issue_eval.prompt_exact_passed_to_model}


def main() -> int:
    args = parse_args()
    evaluator = RandomBenchmarkEvaluator(args)
    return evaluator.run()


if __name__ == "__main__":
    raise SystemExit(main())
