#!/usr/bin/env python3
"""Run GraphRAG queries on mined issues and compute retrieval metrics."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

DEFAULT_RESPONSE_TYPE = "{\"classes\":[{\"path\":\"<java_file_path>.java\"}]}\". No markdown, no explanation, no extra keys."
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
ALLOWED_METHODS = ("local", "drift", "global")

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
    command: List[str]
    status: str
    error: Optional[str]
    raw_response: Optional[str]


@dataclass
class ExpectedIndex:
    targets: Dict[str, ExpectedJavaTarget]
    by_path: Dict[str, str]
    by_simple: Dict[str, Set[str]]
    by_fqcn: Dict[str, Set[str]]


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
    parser.add_argument(
        "mined_file",
        help="Path to a mining JSON file produced by mine_github_issues.py.",
    )
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


def run_graphrag_query(
    graphrag_dir: Path,
    query_text: str,
    method: str,
    data_dir: str,
    timeout_seconds: int,
) -> Tuple[List[str], str]:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "graphrag",
        "query",
        query_text,
        "--root",
        ".",
        "--method",
        method,
        "--data",
        data_dir,
        "--response-type",
        DEFAULT_RESPONSE_TYPE,
    ]
    completed = subprocess.run(
        command,
        cwd=str(graphrag_dir),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    output = completed.stdout.strip()
    if completed.returncode != 0:
        stderr_text = completed.stderr.strip()
        message = stderr_text or output or f"GraphRAG query failed with code {completed.returncode}."
        raise RuntimeError(message)
    return command, output


def default_report_path(mined_file: Path, output_dir: Path) -> Path:
    timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"{mined_file.stem}__graphrag_eval__{timestamp}.json"


def load_mined_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Mined file content must be a JSON object.")
    issues = data.get("issues")
    if not isinstance(issues, list):
        raise ValueError("Mined file must contain an 'issues' list.")
    return data


def evaluate_issue(
    issue: Dict[str, Any],
    graphrag_dir: Path,
    args: argparse.Namespace,
) -> IssueEvaluation:
    expected = build_expected_index(issue)
    issue_number = int(issue.get("number", -1))
    issue_title = str(issue.get("title", "") or "")
    issue_url = str(issue.get("url", "") or "")
    issue_description = issue.get("description_message")

    query = build_query_text(issue_title, issue_description, args.extra_prompt)
    command: List[str] = []
    raw_response: Optional[str] = None
    status = "ok"
    error: Optional[str] = None

    try:
        if args.dry_run:
            predicted_objects: List[PredictedClass] = []
            status = "dry_run"
        else:
            command, raw_response = run_graphrag_query(
                graphrag_dir=graphrag_dir,
                query_text=query,
                method=args.method,
                data_dir=args.data_dir,
                timeout_seconds=args.timeout_seconds,
            )
            predicted_objects = extract_predicted_classes(raw_response, DEFAULT_RESPONSE_TYPE)
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = str(exc)
        predicted_objects = []

    tp, fp, fn, precision, recall, f1, _ = compute_metrics(expected, predicted_objects)
    predicted_classes = sorted(pred.prediction_id for pred in predicted_objects)
    expected_java_files = sorted(target.file_path for target in expected.targets.values())

    return IssueEvaluation(
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
        command=command,
        status=status,
        error=error,
        raw_response=raw_response if args.keep_raw_response else None,
    )


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


def issue_to_json(issue_eval: IssueEvaluation) -> Dict[str, Any]:
    return {
        "issue_number": issue_eval.issue_number,
        "issue_title": issue_eval.issue_title,
        "issue_url": issue_eval.issue_url,
        "status": issue_eval.status,
        "error": issue_eval.error,
        "command": issue_eval.command,
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
        "raw_response": issue_eval.raw_response,
    }


def main() -> int:
    args = parse_args()
    mined_path = Path(args.mined_file).expanduser().resolve()
    if not mined_path.is_file():
        print(f"[ERROR] Mined file not found: {mined_path}", file=sys.stderr)
        return 1

    graphrag_dir = Path(args.graphrag_dir).expanduser().resolve()
    if not graphrag_dir.is_dir() and not args.dry_run:
        print(f"[ERROR] GraphRAG directory not found: {graphrag_dir}", file=sys.stderr)
        return 1
    if not graphrag_dir.is_dir() and args.dry_run:
        print(
            f"[WARN] GraphRAG directory not found (dry-run mode): {graphrag_dir}",
            file=sys.stderr,
        )

    mined_data = load_mined_json(mined_path)
    issues = mined_data.get("issues", [])
    if args.issue_limit:
        issues = issues[: args.issue_limit]

    results: List[IssueEvaluation] = []
    skipped_zero_java = 0
    for issue in issues:
        expected_count = len(build_expected_index(issue).targets)
        if expected_count == 0 and not args.include_empty_java:
            skipped_zero_java += 1
            continue

        issue_number = issue.get("number")
        print(f"[INFO] Evaluating issue #{issue_number}...", file=sys.stderr)
        result = evaluate_issue(issue=issue, graphrag_dir=graphrag_dir, args=args)
        if result.status == "error":
            print(f"[WARN] Issue #{issue_number} failed: {result.error}", file=sys.stderr)
        results.append(result)

    global_metrics = compute_global_metrics(results)

    report = {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "benchmark_source_file": str(mined_path),
        "project_name": mined_data.get("project_name"),
        "github_url": mined_data.get("github_url"),
        "graphrag_dir": str(graphrag_dir),
        "settings": {
            "method": args.method,
            "data_dir": args.data_dir,
            "response_type": DEFAULT_RESPONSE_TYPE,
            "required_pre_prompt": REQUIRED_PRE_PROMPT,
            "extra_prompt": args.extra_prompt,
            "timeout_seconds": args.timeout_seconds,
            "issue_limit": args.issue_limit,
            "output_dir": str(Path(args.output_dir).expanduser().resolve()),
            "dry_run": args.dry_run,
            "include_empty_java": args.include_empty_java,
            "keep_raw_response": args.keep_raw_response,
        },
        "summary": {
            "issues_in_source": len(mined_data.get("issues", [])),
            "issues_considered": len(issues),
            "issues_skipped_no_java_targets": skipped_zero_java,
            **global_metrics,
        },
        "issues": [issue_to_json(item) for item in results],
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_path = (
        Path(args.output_file).expanduser().resolve()
        if args.output_file
        else default_report_path(mined_path, output_dir)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    micro = report["summary"]["micro"]
    print(f"Report written to: {output_path}")
    print(
        "Micro metrics -> "
        f"precision: {micro['precision']:.4f}, "
        f"recall: {micro['recall']:.4f}, "
        f"f1: {micro['f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
