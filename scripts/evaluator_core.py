#!/usr/bin/env python3
"""Shared evaluator core for GraphRAG and LLM API benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
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


def build_expected_index(issue: Dict[str, Any]) -> ExpectedIndex:
    unique_paths: Set[str] = set()
    for pr in issue.get("linked_merged_pull_requests", []):
        for file_path in pr.get("impacted_files", []):
            if isinstance(file_path, str) and file_path.lower().endswith(".java"):
                unique_paths.add(file_path.strip())

    targets: Dict[str, ExpectedJavaTarget] = {}
    by_path: Dict[str, str] = {}
    by_simple: Dict[str, Set[str]] = {}
    by_fqcn: Dict[str, Set[str]] = {}

    for file_path in sorted(unique_paths):
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


def compute_global_metrics(results: List[IssueEvaluation]) -> Dict[str, Any]:
    valid = [r for r in results if r.status != "error"]
    tp = sum(r.true_positives for r in valid)
    fp = sum(r.false_positives for r in valid)
    fn = sum(r.false_negatives for r in valid)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    macro_precision = sum(r.precision for r in valid) / len(valid) if valid else 0.0
    macro_recall = sum(r.recall for r in valid) / len(valid) if valid else 0.0
    macro_f1 = sum(r.f1 for r in valid) / len(valid) if valid else 0.0

    return {
        "issues_evaluated": len(valid),
        "issues_with_errors": len([r for r in results if r.status == "error"]),
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
                )
            )

        global_metrics = compute_global_metrics(results)
        report = {
            "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
            "benchmark_source_file": str(self.mined_path),
            "project_name": mined_data.get("project_name"),
            "github_url": mined_data.get("github_url"),
            "settings": self.settings(),
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
