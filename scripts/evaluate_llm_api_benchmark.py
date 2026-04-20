#!/usr/bin/env python3
"""Evaluate benchmark issues by querying an LLM API (Ollama/OpenAI/Mistral)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from evaluate_graphrag_benchmark import (
    DEFAULT_EVAL_OUTPUT_DIRNAME,
    DEFAULT_ISSUE_PROMPT,
    DEFAULT_RESPONSE_TYPE,
    build_expected_index,
    build_query_text,
    compute_metrics,
    extract_predicted_classes,
    load_mined_json,
)

DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_PROVIDER = "ollama"
PROVIDER_CHOICES = ("ollama", "openai", "mistral")
PROVIDER_DEFAULT_BASE_URL = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com",
    "mistral": "https://api.mistral.ai",
}
PROVIDER_ENV_KEY = {
    "ollama": "OLLAMA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}
DEFAULT_SYSTEM_PROMPT = (
    "You are a Java code localization assistant. "
    "Follow the user output format instructions exactly."
)


class LLMAPIError(RuntimeError):
    """Raised when LLM API call fails."""


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
    prompt_exact_passed_to_llm: str
    status: str
    error: Optional[str]
    raw_response: Optional[str]
    llm_content: Optional[str]
    request_url: Optional[str]
    request_payload: Optional[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_output_dir = project_root / DEFAULT_EVAL_OUTPUT_DIRNAME
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate benchmark issues by querying an LLM API directly "
            "(Ollama/OpenAI/Mistral) and compute precision/recall/F1."
        )
    )
    parser.add_argument("mined_file", help="Path to a mining JSON benchmark file.")
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default=DEFAULT_PROVIDER,
        help=f"LLM provider (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name for the selected provider (e.g. gemma2, gpt-4.1-mini, mistral-small-latest).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional custom provider base URL. If omitted, provider default is used.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key (fallback to provider env var).",
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
        help=f"Timeout in seconds for each LLM API call (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--extra-prompt",
        default=DEFAULT_ISSUE_PROMPT,
        help="Prompt suffix appended after title+description.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature sent to the LLM API (default: 0.0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens for completion.",
    )
    parser.add_argument(
        "--include-empty-java",
        action="store_true",
        help="Evaluate issues even when no Java files are expected (default: skip).",
    )
    parser.add_argument(
        "--keep-raw-response",
        action="store_true",
        help="Store full raw LLM response JSON in output report.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt sent to the LLM API.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and evaluate structure, but do not call the LLM API.",
    )
    return parser.parse_args()


def resolve_api_key(provider: str, cli_api_key: Optional[str]) -> Optional[str]:
    if cli_api_key and cli_api_key.strip():
        return cli_api_key.strip()
    env_key = PROVIDER_ENV_KEY.get(provider)
    if not env_key:
        return None
    value = os.getenv(env_key)
    return value.strip() if value and value.strip() else None


def resolve_base_url(provider: str, cli_base_url: Optional[str]) -> str:
    if cli_base_url and cli_base_url.strip():
        return cli_base_url.strip()
    return PROVIDER_DEFAULT_BASE_URL[provider]


def build_chat_completions_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if not parsed.scheme:
        base_url = f"http://{base_url}"
    cleaned = base_url.rstrip("/")
    lowered = cleaned.lower()
    if lowered.endswith("/chat/completions"):
        return cleaned
    if lowered.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def extract_message_content(api_response: Dict[str, Any]) -> str:
    choices = api_response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts: List[str] = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                    if parts:
                        return "\n".join(parts)
            text = first.get("text")
            if isinstance(text, str):
                return text
    raise LLMAPIError("LLM response does not contain a readable message content.")


def call_chat_completions_api(
    url: str,
    api_key: Optional[str],
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: Optional[int],
    timeout_seconds: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    payload_json = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "benchmark-graphrag-evaluator",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url=url, data=payload_json, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise LLMAPIError(f"HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise LLMAPIError(str(error.reason)) from error

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        raise LLMAPIError(f"Invalid JSON response: {body[:1000]}") from error

    content = extract_message_content(parsed)
    return payload, parsed, content


def safe_model_for_filename(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model.strip().lower()) or "model"


def default_report_path(mined_file: Path, output_dir: Path, provider: str, model: str) -> Path:
    timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / (
        f"{mined_file.stem}__{provider}__{safe_model_for_filename(model)}__llm_eval__{timestamp}.json"
    )


def evaluate_issue(
    issue: Dict[str, Any],
    user_prompt: str,
    args: argparse.Namespace,
    request_url: Optional[str],
    api_key: Optional[str],
) -> IssueEvaluation:
    expected = build_expected_index(issue)
    issue_number = int(issue.get("number", -1))
    issue_title = str(issue.get("title", "") or "")
    issue_url = str(issue.get("url", "") or "")

    predicted_objects = []
    raw_response: Optional[str] = None
    llm_content: Optional[str] = None
    request_payload: Optional[Dict[str, Any]] = None
    status = "ok"
    error: Optional[str] = None

    try:
        if args.dry_run:
            status = "dry_run"
        else:
            request_payload, response_json, llm_content = call_chat_completions_api(
                url=request_url or "",
                api_key=api_key,
                model=args.model,
                system_prompt=args.system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
            )
            raw_response = json.dumps(response_json, ensure_ascii=False)
            predicted_objects = extract_predicted_classes(llm_content, DEFAULT_RESPONSE_TYPE)
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
        query=user_prompt,
        prompt_exact_passed_to_llm=user_prompt,
        status=status,
        error=error,
        raw_response=raw_response if args.keep_raw_response else None,
        llm_content=llm_content,
        request_url=request_url,
        request_payload=request_payload,
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
        "request_url": issue_eval.request_url,
        "request_payload": issue_eval.request_payload,
        "query": issue_eval.query,
        "prompt_exact_passed_to_llm": issue_eval.prompt_exact_passed_to_llm,
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
        "llm_content": issue_eval.llm_content,
        "raw_response": issue_eval.raw_response,
    }


def main() -> int:
    args = parse_args()
    mined_path = Path(args.mined_file).expanduser().resolve()
    if not mined_path.is_file():
        print(f"[ERROR] Mined file not found: {mined_path}", file=sys.stderr)
        return 1

    api_key = resolve_api_key(args.provider, args.api_key)
    if args.provider in {"openai", "mistral"} and not api_key and not args.dry_run:
        env_key = PROVIDER_ENV_KEY[args.provider]
        print(
            f"[ERROR] Missing API key for provider '{args.provider}'. Use --api-key or set {env_key}.",
            file=sys.stderr,
        )
        return 1

    base_url = resolve_base_url(args.provider, args.base_url)
    request_url = build_chat_completions_url(base_url) if not args.dry_run else None

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
        title = str(issue.get("title", "") or "")
        description = issue.get("description_message")
        prompt = build_query_text(title, description, args.extra_prompt)

        print(f"[INFO] Evaluating issue #{issue_number}...", file=sys.stderr)
        result = evaluate_issue(issue, prompt, args, request_url, api_key)
        if result.status == "error":
            print(f"[WARN] Issue #{issue_number} failed: {result.error}", file=sys.stderr)
        results.append(result)

    global_metrics = compute_global_metrics(results)

    report = {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "benchmark_source_file": str(mined_path),
        "project_name": mined_data.get("project_name"),
        "github_url": mined_data.get("github_url"),
        "settings": {
            "provider": args.provider,
            "model": args.model,
            "base_url": base_url,
            "request_url": request_url,
            "response_type": DEFAULT_RESPONSE_TYPE,
            "extra_prompt": args.extra_prompt,
            "system_prompt": args.system_prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
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
        else default_report_path(mined_path, output_dir, args.provider, args.model)
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
