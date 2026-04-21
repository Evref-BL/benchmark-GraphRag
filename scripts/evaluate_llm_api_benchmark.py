#!/usr/bin/env python3
"""Evaluate benchmark issues via LLM API (specialization of BaseBenchmarkEvaluator)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

from evaluator_core import (
    BaseBenchmarkEvaluator,
    DEFAULT_EVAL_OUTPUT_DIRNAME,
    DEFAULT_ISSUE_PROMPT,
    DEFAULT_RESPONSE_TYPE,
    DEFAULT_TIMEOUT_SECONDS,
    PredictionResult,
    extract_predicted_classes,
)

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
    """Raised when an LLM API call fails."""


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
                    parts = []
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


class LLMAPIEvaluator(BaseBenchmarkEvaluator):
    """LLM API specialization of the shared evaluator flow."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.provider = args.provider
        self.model = args.model
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.timeout_seconds = args.timeout_seconds
        self.system_prompt = args.system_prompt
        self.base_url = resolve_base_url(args.provider, args.base_url)
        self.request_url = build_chat_completions_url(self.base_url) if not self.dry_run else None
        self.api_key = resolve_api_key(args.provider, args.api_key)

    def evaluator_label(self) -> str:
        return f"llm_api:{self.provider}"

    def validate_runtime(self) -> None:
        if self.dry_run:
            return
        if self.provider in {"openai", "mistral"} and not self.api_key:
            env_key = PROVIDER_ENV_KEY[self.provider]
            raise RuntimeError(
                f"Missing API key for provider '{self.provider}'. Use --api-key or set {env_key}."
            )

    def predict_for_issue(self, issue: Dict[str, Any], query: str) -> PredictionResult:
        payload, response_json, content = call_chat_completions_api(
            url=self.request_url or "",
            api_key=self.api_key,
            model=self.model,
            system_prompt=self.system_prompt,
            user_prompt=query,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
        )
        predicted = extract_predicted_classes(content, DEFAULT_RESPONSE_TYPE)
        return PredictionResult(
            predicted_objects=predicted,
            raw_response=json.dumps(response_json, ensure_ascii=False),
            llm_content=content,
            request_url=self.request_url,
            request_payload=payload,
        )

    def default_report_path(self, mined_path: Path, output_dir: Path) -> Path:
        timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return output_dir / (
            f"{mined_path.stem}__{self.provider}__{safe_model_for_filename(self.model)}__llm_eval__{timestamp}.json"
        )

    def settings(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "request_url": self.request_url,
            "response_type": DEFAULT_RESPONSE_TYPE,
            "extra_prompt": self.extra_prompt,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "issue_limit": self.issue_limit,
            "output_dir": str(self.output_dir),
            "dry_run": self.dry_run,
            "include_empty_java": self.include_empty_java,
            "keep_raw_response": self.keep_raw_response,
        }

    def issue_extra_fields(self, issue_eval) -> Dict[str, Any]:
        return {"prompt_exact_passed_to_llm": issue_eval.prompt_exact_passed_to_model}


def main() -> int:
    args = parse_args()
    evaluator = LLMAPIEvaluator(args)
    return evaluator.run()


if __name__ == "__main__":
    raise SystemExit(main())
