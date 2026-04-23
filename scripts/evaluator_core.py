#!/usr/bin/env python3
"""Shared evaluator core for GraphRAG and LLM API benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
import datetime as dt
import json
from dataclasses import dataclass
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

DEFAULT_RESPONSE_TYPE = (
    "{\"classes\":[{\"path\":\"<java_file_path>.java\"}]}. "
    "No markdown, no explanation, no extra keys."
)
REQUIRED_PRE_PROMPT = (
    "MANDATORY OUTPUT FORMAT:\n"
    "Follow STRICTLY the response_type specification below:\n"
    f"{DEFAULT_RESPONSE_TYPE}\n"
    "No markdown and no extra text outside this format."
)
DEFAULT_ISSUE_PROMPT = (
    "Identify the Java class file paths impacted by the issue resolution. "
    "Use only paths ending with .java in the required JSON schema."
)
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_EVAL_OUTPUT_DIRNAME = "evaluation_results"
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 42

FQCN_RE = re.compile(r"\b(?:[a-z_][a-z0-9_$]*\.)+[A-Z][A-Za-z0-9_$]*\b")
JAVA_PATH_RE = re.compile(r"\b(?:[A-Za-z0-9_.-]+/)*[A-Za-z_][A-Za-z0-9_$]*\.java\b")
MARKDOWN_CODE_RE = re.compile(r"`([^`]+)`")
SIMPLE_CLASS_RE = re.compile(r"^[A-Z][A-Za-z0-9_$]*$")
LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|(?:\d+[\.\)]))\s*")
JSON_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ExpectedJavaTarget:
    file_path: str
    file_key: str
    simple_key: str
    fqcn_keys: Set[str]


@dataclass(frozen=True)
class PredictedClass:
    prediction_id: str
    raw: str
    path_key: Optional[str]
    simple_key: Optional[str]
    fqcn_key: Optional[str]


@dataclass
class ExpectedIndex:
    targets: Dict[str, ExpectedJavaTarget]
    by_path: Dict[str, str]
    by_simple: Dict[str, Set[str]]
    by_fqcn: Dict[str, Set[str]]


@dataclass
class PredictionResult:
    predicted_objects: List[PredictedClass]
    raw_response: Optional[str] = None
    command: Optional[List[str]] = None
    llm_content: Optional[str] = None
    request_url: Optional[str] = None
    request_payload: Optional[Dict[str, Any]] = None


@dataclass
class IssueEvaluation:
    issue_number: int
    issue_title: str
    issue_url: str
    expected_java_files: List[str]
    predicted_classes: List[str]
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    query: str
    prompt_exact_passed_to_model: str
    status: str
    error: Optional[str]
    raw_response: Optional[str] = None
    command: Optional[List[str]] = None
    llm_content: Optional[str] = None
    request_url: Optional[str] = None
    request_payload: Optional[Dict[str, Any]] = None
    expected_java_files_in_repo: Optional[List[str]] = None
    true_positives_in_repo: Optional[int] = None
    false_positives_in_repo: Optional[int] = None
    false_negatives_in_repo: Optional[int] = None
    precision_in_repo: Optional[float] = None
    recall_in_repo: Optional[float] = None
    f1_in_repo: Optional[float] = None


class EvaluationStopped(Exception):
    """Raised by interactive evaluators to stop early and persist partial results."""


def normalize_path(path: str) -> str:
    return path.strip().replace("\\", "/").strip("/").lower()


def java_path_to_fqcn_candidates(java_file_path: str) -> Set[str]:
    path = java_file_path.strip().replace("\\", "/")
    if not path.lower().endswith(".java"):
        return set()

    no_ext = path[:-5].strip("/")
    candidates: Set[str] = set()
    markers = (
        "/src/main/java/",
        "/src/test/java/",
        "/main/java/",
        "/test/java/",
        "/java/",
    )
    lowered = f"/{no_ext.lower()}/"
    original = f"/{no_ext}/"
    for marker in markers:
        idx = lowered.find(marker)
        if idx != -1:
            start = idx + len(marker)
            rel = original[start:-1]
            if rel:
                candidates.add(rel.replace("/", ".").strip(".").lower())

    if "/" in no_ext:
        candidates.add(no_ext.replace("/", ".").strip(".").lower())
    return {c for c in candidates if c}


def build_expected_index_from_paths(java_paths: Iterable[str]) -> ExpectedIndex:
    """Build a matchable index from a list of Java file paths."""
    unique_paths = sorted(
        {
            file_path.strip()
            for file_path in java_paths
            if isinstance(file_path, str) and file_path.strip().lower().endswith(".java")
        }
    )
    targets: Dict[str, ExpectedJavaTarget] = {}
    by_path: Dict[str, str] = {}
    by_simple: Dict[str, Set[str]] = {}
    by_fqcn: Dict[str, Set[str]] = {}

    for file_path in unique_paths:
        file_key = normalize_path(file_path)
        if not file_key:
            continue
        simple_key = Path(file_path).stem.lower()
        fqcn_keys = java_path_to_fqcn_candidates(file_path)
        target = ExpectedJavaTarget(
            file_path=file_path,
            file_key=file_key,
            simple_key=simple_key,
            fqcn_keys=fqcn_keys,
        )
        targets[file_key] = target
        by_path[file_key] = file_key
        by_simple.setdefault(simple_key, set()).add(file_key)
        for fqcn in fqcn_keys:
            by_fqcn.setdefault(fqcn, set()).add(file_key)

    return ExpectedIndex(
        targets=targets,
        by_path=by_path,
        by_simple=by_simple,
        by_fqcn=by_fqcn,
    )


def build_expected_index(issue: Dict[str, Any]) -> ExpectedIndex:
    """Build the expected index for one benchmark issue."""
    unique_paths: Set[str] = set()
    for pr in issue.get("linked_merged_pull_requests", []):
        for file_path in pr.get("impacted_files", []):
            if isinstance(file_path, str) and file_path.lower().endswith(".java"):
                unique_paths.add(file_path.strip())
    return build_expected_index_from_paths(unique_paths)


def filter_expected_index_by_file_keys(
    expected: ExpectedIndex, allowed_file_keys: Set[str]
) -> ExpectedIndex:
    """Keep only expected targets that belong to an allowed normalized path set."""
    filtered_paths = [
        target.file_path for file_key, target in expected.targets.items() if file_key in allowed_file_keys
    ]
    return build_expected_index_from_paths(filtered_paths)


def collect_repository_file_keys(project_root: Path, extension: str = ".java") -> Set[str]:
    """Collect normalized relative file keys for a given extension under a project root."""
    normalized_ext = extension.strip().lower()
    if not normalized_ext:
        raise ValueError("extension cannot be empty.")
    if not normalized_ext.startswith("."):
        normalized_ext = f".{normalized_ext}"

    keys: Set[str] = set()
    for file_path in project_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() != normalized_ext:
            continue
        rel_path = file_path.relative_to(project_root).as_posix()
        keys.add(normalize_path(rel_path))
    return keys


def build_query_text(title: str, description: Optional[str], extra_prompt: str) -> str:
    safe_title = (title or "").strip()
    safe_desc = (description or "").strip() or "(no description provided)"
    prompt = (extra_prompt or "").strip()
    return (
        f"Issue title:\n{safe_title}\n\n"
        f"Issue description:\n{safe_desc}\n\n"
        f"Required pre-prompt:\n{REQUIRED_PRE_PROMPT}\n\n"
        f"Task:\n{prompt}"
    )


def normalize_prediction_token(raw: str) -> Optional[PredictedClass]:
    token = raw.strip().strip("`").strip()
    token = re.sub(r"^\[([^\]]+)\]\([^)]+\)$", r"\1", token)
    token = token.strip(" \t\r\n,;:.")
    if not token:
        return None

    java_match = JAVA_PATH_RE.search(token)
    if java_match:
        path = java_match.group(0)
        path_key = normalize_path(path)
        simple_key = Path(path).stem.lower()
        fqcn_keys = sorted(java_path_to_fqcn_candidates(path))
        fqcn_key = fqcn_keys[0] if fqcn_keys else None
        return PredictedClass(
            prediction_id=f"path:{path_key}",
            raw=raw,
            path_key=path_key,
            simple_key=simple_key or None,
            fqcn_key=fqcn_key,
        )

    fqcn_match = FQCN_RE.search(token)
    if fqcn_match:
        fqcn = fqcn_match.group(0)
        return PredictedClass(
            prediction_id=f"fqcn:{fqcn.lower()}",
            raw=raw,
            path_key=None,
            simple_key=fqcn.split(".")[-1].lower(),
            fqcn_key=fqcn.lower(),
        )

    simple = token.split()[-1]
    simple = simple.strip(" \t\r\n,;:.")
    if SIMPLE_CLASS_RE.fullmatch(simple):
        simple_key = simple.lower()
        return PredictedClass(
            prediction_id=f"simple:{simple_key}",
            raw=raw,
            path_key=None,
            simple_key=simple_key,
            fqcn_key=None,
        )
    return None


def extract_predicted_classes_generic(text: str) -> List[PredictedClass]:
    candidates: List[PredictedClass] = []
    seen_ids: Set[str] = set()

    tokens: List[str] = []
    for line in text.splitlines():
        cleaned_line = LINE_PREFIX_RE.sub("", line).strip()
        if not cleaned_line:
            continue
        tokens.append(cleaned_line)
        for part in re.split(r"[;,]", cleaned_line):
            part = part.strip()
            if part:
                tokens.append(part)

    tokens.extend(MARKDOWN_CODE_RE.findall(text))
    tokens.extend(JAVA_PATH_RE.findall(text))
    tokens.extend(FQCN_RE.findall(text))

    for token in tokens:
        parsed = normalize_prediction_token(token)
        if not parsed:
            continue
        if parsed.prediction_id in seen_ids:
            continue
        seen_ids.add(parsed.prediction_id)
        candidates.append(parsed)
    return candidates


def is_json_response_type(response_type: str) -> bool:
    lowered = response_type.lower()
    return "json" in lowered or ("{" in response_type and "}" in response_type)


def parse_json_payload_from_response(text: str) -> Optional[Any]:
    candidates: List[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    for match in JSON_CODEBLOCK_RE.finditer(text):
        block = match.group(1).strip()
        if block:
            candidates.append(block)

    first_curly = text.find("{")
    last_curly = text.rfind("}")
    if 0 <= first_curly < last_curly:
        candidates.append(text[first_curly : last_curly + 1].strip())

    seen: Set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def collect_json_class_like_values(payload: Any) -> List[str]:
    values: List[str] = []
    if isinstance(payload, dict):
        classes = payload.get("classes")
        if isinstance(classes, list):
            for entry in classes:
                if isinstance(entry, str):
                    values.append(entry)
                elif isinstance(entry, dict):
                    for key in ("path", "class_path", "file", "name", "class"):
                        item_value = entry.get(key)
                        if isinstance(item_value, str):
                            values.append(item_value)

        for key in ("paths", "java_files", "files"):
            entry = payload.get(key)
            if isinstance(entry, list):
                values.extend(v for v in entry if isinstance(v, str))
            elif isinstance(entry, str):
                values.append(entry)
    elif isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, str):
                values.append(entry)
            elif isinstance(entry, dict):
                for key in ("path", "class_path", "file", "name", "class"):
                    item_value = entry.get(key)
                    if isinstance(item_value, str):
                        values.append(item_value)
    return values


def extract_predicted_classes_from_json_payload(payload: Any) -> List[PredictedClass]:
    seen_ids: Set[str] = set()
    classes: List[PredictedClass] = []
    for raw_value in collect_json_class_like_values(payload):
        parsed = normalize_prediction_token(raw_value)
        if not parsed:
            continue
        if parsed.prediction_id in seen_ids:
            continue
        seen_ids.add(parsed.prediction_id)
        classes.append(parsed)
    return classes


def extract_predicted_classes(text: str, response_type: str) -> List[PredictedClass]:
    if is_json_response_type(response_type):
        payload = parse_json_payload_from_response(text)
        if payload is not None:
            extracted = extract_predicted_classes_from_json_payload(payload)
            if extracted:
                return extracted
    return extract_predicted_classes_generic(text)


def pick_target_for_prediction(pred: PredictedClass, expected: ExpectedIndex) -> Optional[str]:
    if pred.path_key and pred.path_key in expected.by_path:
        return expected.by_path[pred.path_key]

    if pred.fqcn_key and pred.fqcn_key in expected.by_fqcn:
        targets = expected.by_fqcn[pred.fqcn_key]
        if len(targets) == 1:
            return next(iter(targets))

    if pred.simple_key and pred.simple_key in expected.by_simple:
        targets = expected.by_simple[pred.simple_key]
        if len(targets) == 1:
            return next(iter(targets))
    return None


def compute_metrics(
    expected: ExpectedIndex, predicted: Iterable[PredictedClass]
) -> Tuple[int, int, int, float, float, float, List[str]]:
    matched_targets: Set[str] = set()
    unmatched_predictions: Set[str] = set()

    for pred in predicted:
        target = pick_target_for_prediction(pred, expected)
        if target:
            matched_targets.add(target)
        else:
            unmatched_predictions.add(pred.prediction_id)

    tp = len(matched_targets)
    fp = len(unmatched_predictions)
    fn = max(0, len(expected.targets) - tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(expected.targets) if expected.targets else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    matched_files = sorted(expected.targets[file_key].file_path for file_key in matched_targets)
    return tp, fp, fn, precision, recall, f1, matched_files


def _empty_micro_macro() -> Dict[str, Any]:
    return {
        "micro": {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
        "macro": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
    }


def _aggregate_rows(rows: List[Tuple[int, int, int, float, float, float]]) -> Dict[str, Any]:
    if not rows:
        return {"issues_evaluated": 0, **_empty_micro_macro()}

    tp = sum(row[0] for row in rows)
    fp = sum(row[1] for row in rows)
    fn = sum(row[2] for row in rows)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    macro_precision = sum(row[3] for row in rows) / len(rows)
    macro_recall = sum(row[4] for row in rows) / len(rows)
    macro_f1 = sum(row[5] for row in rows) / len(rows)

    return {
        "issues_evaluated": len(rows),
        "micro": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
    }


def _percentile(values: List[float], quantile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * quantile
    lower_idx = int(math.floor(position))
    upper_idx = int(math.ceil(position))
    if lower_idx == upper_idx:
        return sorted_values[lower_idx]
    weight = position - lower_idx
    return sorted_values[lower_idx] * (1.0 - weight) + sorted_values[upper_idx] * weight


def _bootstrap_ci95(
    rows: List[Tuple[int, int, int, float, float, float]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Compute 95% bootstrap confidence intervals for micro/macro metrics."""
    if not rows:
        return None

    samples = max(1, int(bootstrap_samples))
    rng = random.Random(bootstrap_seed)
    n = len(rows)

    micro_precision_values: List[float] = []
    micro_recall_values: List[float] = []
    micro_f1_values: List[float] = []
    macro_precision_values: List[float] = []
    macro_recall_values: List[float] = []
    macro_f1_values: List[float] = []

    for _ in range(samples):
        sampled_rows = [rows[rng.randrange(n)] for _ in range(n)]
        aggregated = _aggregate_rows(sampled_rows)
        micro = aggregated["micro"]
        macro = aggregated["macro"]
        micro_precision_values.append(float(micro["precision"]))
        micro_recall_values.append(float(micro["recall"]))
        micro_f1_values.append(float(micro["f1"]))
        macro_precision_values.append(float(macro["precision"]))
        macro_recall_values.append(float(macro["recall"]))
        macro_f1_values.append(float(macro["f1"]))

    def ci(values: List[float]) -> Dict[str, float]:
        return {
            "lower": _percentile(values, 0.025),
            "upper": _percentile(values, 0.975),
        }

    return {
        "bootstrap_samples": samples,
        "bootstrap_seed": bootstrap_seed,
        "micro": {
            "precision": ci(micro_precision_values),
            "recall": ci(micro_recall_values),
            "f1": ci(micro_f1_values),
        },
        "macro": {
            "precision": ci(macro_precision_values),
            "recall": ci(macro_recall_values),
            "f1": ci(macro_f1_values),
        },
    }


def compute_global_metrics(
    results: List[IssueEvaluation],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: Optional[int] = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Aggregate global metrics + IC95%, with an optional in-repo-only view.

    `in_repo_only` is computed only when issue-level in-repo metrics are available.
    For fairness, global in-repo aggregates include only issues that still have at least
    one expected target inside the local repository universe.
    """
    valid = [r for r in results if r.status != "error"]
    overall_rows = [
        (r.true_positives, r.false_positives, r.false_negatives, r.precision, r.recall, r.f1) for r in valid
    ]
    overall = _aggregate_rows(overall_rows)
    overall_ci95 = _bootstrap_ci95(
        overall_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )

    in_repo_candidates = [
        r
        for r in valid
        if r.true_positives_in_repo is not None
        and r.false_positives_in_repo is not None
        and r.false_negatives_in_repo is not None
        and r.precision_in_repo is not None
        and r.recall_in_repo is not None
        and r.f1_in_repo is not None
    ]
    in_repo_with_targets = [
        r for r in in_repo_candidates if (r.expected_java_files_in_repo and len(r.expected_java_files_in_repo) > 0)
    ]
    in_repo_rows = [
        (
            int(r.true_positives_in_repo),
            int(r.false_positives_in_repo),
            int(r.false_negatives_in_repo),
            float(r.precision_in_repo),
            float(r.recall_in_repo),
            float(r.f1_in_repo),
        )
        for r in in_repo_with_targets
    ]
    in_repo = _aggregate_rows(in_repo_rows)
    in_repo_ci95 = _bootstrap_ci95(
        in_repo_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=(None if bootstrap_seed is None else bootstrap_seed + 1),
    )

    return {
        "issues_evaluated": overall["issues_evaluated"],
        "issues_with_errors": len([r for r in results if r.status == "error"]),
        "micro": overall["micro"],
        "macro": overall["macro"],
        "confidence_interval_95": overall_ci95,
        "in_repo_only": {
            "enabled": len(in_repo_candidates) > 0,
            "issues_with_in_repo_metrics": len(in_repo_candidates),
            "issues_with_in_repo_targets": len(in_repo_with_targets),
            "issues_without_in_repo_targets": len(in_repo_candidates) - len(in_repo_with_targets),
            "issues_evaluated": in_repo["issues_evaluated"],
            "micro": in_repo["micro"],
            "macro": in_repo["macro"],
            "confidence_interval_95": in_repo_ci95,
        },
    }


def load_mined_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Mined file content must be a JSON object.")
    issues = data.get("issues")
    if not isinstance(issues, list):
        raise ValueError("Mined file must contain an 'issues' list.")
    return data


class BaseBenchmarkEvaluator(ABC):
    """Shared sequential evaluator loop; subclasses only implement inference specifics."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mined_path = Path(args.mined_file).expanduser().resolve()
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_file = (
            Path(args.output_file).expanduser().resolve() if getattr(args, "output_file", None) else None
        )
        self.extra_prompt = getattr(args, "extra_prompt", DEFAULT_ISSUE_PROMPT)
        self.issue_limit = getattr(args, "issue_limit", None)
        self.include_empty_java = bool(getattr(args, "include_empty_java", False))
        self.keep_raw_response = bool(getattr(args, "keep_raw_response", False))
        self.dry_run = bool(getattr(args, "dry_run", False))
        self.bootstrap_samples = max(
            1, int(getattr(args, "bootstrap_samples", DEFAULT_BOOTSTRAP_SAMPLES))
        )
        self.bootstrap_seed = getattr(args, "bootstrap_seed", DEFAULT_BOOTSTRAP_SEED)
        self.last_output_path: Optional[Path] = None
        self.last_report: Optional[Dict[str, Any]] = None

    @abstractmethod
    def evaluator_label(self) -> str:
        """Human-readable evaluator label."""

    @abstractmethod
    def validate_runtime(self) -> None:
        """Validate runtime prerequisites before evaluating."""

    @abstractmethod
    def predict_for_issue(self, issue: Dict[str, Any], query: str) -> PredictionResult:
        """Produce predictions for one issue."""

    @abstractmethod
    def default_report_path(self, mined_path: Path, output_dir: Path) -> Path:
        """Build default output report path."""

    @abstractmethod
    def settings(self) -> Dict[str, Any]:
        """Return evaluator-specific settings for report JSON."""

    def issue_extra_fields(self, issue_eval: IssueEvaluation) -> Dict[str, Any]:
        """Hook for evaluator-specific issue fields in report JSON."""
        return {}

    def in_repo_reference_keys(self) -> Optional[Set[str]]:
        """Optional normalized path universe used to compute in_repo_only metrics."""
        return None

    def _issue_to_json(self, issue_eval: IssueEvaluation) -> Dict[str, Any]:
        payload = {
            "issue_number": issue_eval.issue_number,
            "issue_title": issue_eval.issue_title,
            "issue_url": issue_eval.issue_url,
            "status": issue_eval.status,
            "error": issue_eval.error,
            "query": issue_eval.query,
            "expected_java_files": issue_eval.expected_java_files,
            "expected_java_files_count": len(issue_eval.expected_java_files),
            "predicted_classes": issue_eval.predicted_classes,
            "predicted_classes_count": len(issue_eval.predicted_classes),
            "true_positives": issue_eval.true_positives,
            "false_positives": issue_eval.false_positives,
            "false_negatives": issue_eval.false_negatives,
            "precision": issue_eval.precision,
            "recall": issue_eval.recall,
            "f1": issue_eval.f1,
            "command": issue_eval.command,
            "request_url": issue_eval.request_url,
            "request_payload": issue_eval.request_payload,
            "llm_content": issue_eval.llm_content,
            "raw_response": issue_eval.raw_response,
            "in_repo_only": None,
        }
        if issue_eval.expected_java_files_in_repo is not None:
            payload["in_repo_only"] = {
                "expected_java_files": issue_eval.expected_java_files_in_repo,
                "expected_java_files_count": len(issue_eval.expected_java_files_in_repo),
                "true_positives": issue_eval.true_positives_in_repo,
                "false_positives": issue_eval.false_positives_in_repo,
                "false_negatives": issue_eval.false_negatives_in_repo,
                "precision": issue_eval.precision_in_repo,
                "recall": issue_eval.recall_in_repo,
                "f1": issue_eval.f1_in_repo,
            }
        payload.update(self.issue_extra_fields(issue_eval))
        return payload

    def run(self) -> int:
        if not self.mined_path.is_file():
            print(f"[ERROR] Mined file not found: {self.mined_path}", file=sys.stderr)
            return 1

        try:
            self.validate_runtime()
        except RuntimeError as error:
            print(f"[ERROR] {error}", file=sys.stderr)
            return 1

        in_repo_keys: Optional[Set[str]] = self.in_repo_reference_keys()
        if in_repo_keys is not None:
            print(
                "[INFO] in_repo_only enabled with reference pool size: "
                f"{len(in_repo_keys)} files",
                file=sys.stderr,
            )

        mined_data = load_mined_json(self.mined_path)
        issues = mined_data.get("issues", [])
        if self.issue_limit:
            issues = issues[: self.issue_limit]

        results: List[IssueEvaluation] = []
        skipped_zero_java = 0
        stopped_early = False
        stop_reason: Optional[str] = None

        for issue in issues:
            expected = build_expected_index(issue)
            if len(expected.targets) == 0 and not self.include_empty_java:
                skipped_zero_java += 1
                continue

            issue_number = int(issue.get("number", -1))
            issue_title = str(issue.get("title", "") or "")
            issue_url = str(issue.get("url", "") or "")
            description = issue.get("description_message")
            query = build_query_text(issue_title, description, self.extra_prompt)

            print(f"[INFO] Evaluating issue #{issue_number}...", file=sys.stderr)

            status = "ok"
            error: Optional[str] = None
            prediction = PredictionResult(predicted_objects=[])
            try:
                if self.dry_run:
                    status = "dry_run"
                else:
                    prediction = self.predict_for_issue(issue, query)
            except EvaluationStopped as exc:
                stopped_early = True
                stop_reason = str(exc).strip() or "Stopped by user."
                print(
                    f"[INFO] Evaluation stopped by user before issue #{issue_number} completion.",
                    file=sys.stderr,
                )
                break
            except Exception as exc:  # noqa: BLE001
                status = "error"
                error = str(exc)
                prediction = PredictionResult(predicted_objects=[])

            if status == "error":
                print(f"[WARN] Issue #{issue_number} failed: {error}", file=sys.stderr)

            tp, fp, fn, precision, recall, f1, _ = compute_metrics(expected, prediction.predicted_objects)
            predicted_classes = sorted(pred.prediction_id for pred in prediction.predicted_objects)
            expected_java_files = sorted(target.file_path for target in expected.targets.values())
            expected_java_files_in_repo: Optional[List[str]] = None
            tp_in_repo: Optional[int] = None
            fp_in_repo: Optional[int] = None
            fn_in_repo: Optional[int] = None
            precision_in_repo: Optional[float] = None
            recall_in_repo: Optional[float] = None
            f1_in_repo: Optional[float] = None

            if in_repo_keys is not None:
                # Build a second expected set restricted to files that are truly available
                # in the local repository, then compute an in-repo-only metric view.
                expected_in_repo = filter_expected_index_by_file_keys(expected, in_repo_keys)
                expected_java_files_in_repo = sorted(
                    target.file_path for target in expected_in_repo.targets.values()
                )
                (
                    tp_in_repo,
                    fp_in_repo,
                    fn_in_repo,
                    precision_in_repo,
                    recall_in_repo,
                    f1_in_repo,
                    _,
                ) = compute_metrics(expected_in_repo, prediction.predicted_objects)

            results.append(
                IssueEvaluation(
                    issue_number=issue_number,
                    issue_title=issue_title,
                    issue_url=issue_url,
                    expected_java_files=expected_java_files,
                    predicted_classes=predicted_classes,
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=fn,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    query=query,
                    prompt_exact_passed_to_model=query,
                    status=status,
                    error=error,
                    raw_response=prediction.raw_response if self.keep_raw_response else None,
                    command=prediction.command,
                    llm_content=prediction.llm_content,
                    request_url=prediction.request_url,
                    request_payload=prediction.request_payload,
                    expected_java_files_in_repo=expected_java_files_in_repo,
                    true_positives_in_repo=tp_in_repo,
                    false_positives_in_repo=fp_in_repo,
                    false_negatives_in_repo=fn_in_repo,
                    precision_in_repo=precision_in_repo,
                    recall_in_repo=recall_in_repo,
                    f1_in_repo=f1_in_repo,
                )
            )

        global_metrics = compute_global_metrics(
            results,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_seed=self.bootstrap_seed,
        )
        report_settings = self.settings().copy()
        report_settings.update(
            {
                "bootstrap_samples": self.bootstrap_samples,
                "bootstrap_seed": self.bootstrap_seed,
                "in_repo_only_enabled": in_repo_keys is not None,
                "in_repo_only_reference_pool_size": (len(in_repo_keys) if in_repo_keys is not None else None),
            }
        )
        report = {
            "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
            "benchmark_source_file": str(self.mined_path),
            "project_name": mined_data.get("project_name"),
            "github_url": mined_data.get("github_url"),
            "settings": report_settings,
            "summary": {
                "issues_in_source": len(mined_data.get("issues", [])),
                "issues_considered": len(issues),
                "issues_skipped_no_java_targets": skipped_zero_java,
                "stopped_early_by_user": stopped_early,
                "stop_reason": stop_reason,
                **global_metrics,
            },
            "issues": [self._issue_to_json(item) for item in results],
        }

        output_path = self.output_file or self.default_report_path(self.mined_path, self.output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
        self.last_output_path = output_path
        self.last_report = report

        micro = report["summary"]["micro"]
        print(f"Report written to: {output_path}")
        print(
            "Micro metrics -> "
            f"precision: {micro['precision']:.4f}, "
            f"recall: {micro['recall']:.4f}, "
            f"f1: {micro['f1']:.4f}"
        )
        return 0
