#!/usr/bin/env python3
"""Export LLM queries from a benchmark file with issue metadata and expected targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, List

from evaluator_core import (
    DEFAULT_ISSUE_PROMPT,
    build_expected_index,
    build_query_text,
)

DEFAULT_QUERIES_DIRNAME = "queries"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_output_dir = project_root / DEFAULT_QUERIES_DIRNAME
    parser = argparse.ArgumentParser(
        description=(
            "Export benchmark issues as LLM queries only, using the same query builder "
            "as evaluate_graphrag_benchmark.py."
        )
    )
    parser.add_argument(
        "benchmark_file",
        help="Path to a benchmark JSON file produced by mine_github_issues.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help=f"Output directory for queries JSON files (default: project-root/{DEFAULT_QUERIES_DIRNAME}).",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional full output file path (overrides --output-dir).",
    )
    parser.add_argument(
        "--issue-limit",
        type=int,
        default=None,
        help="Optional cap on number of issues exported.",
    )
    parser.add_argument(
        "--extra-prompt",
        default=DEFAULT_ISSUE_PROMPT,
        help="Prompt suffix appended after title+description.",
    )
    parser.add_argument(
        "--include-empty-java",
        action="store_true",
        help="Include issues with no Java targets (default: skip, like evaluator).",
    )
    return parser.parse_args()


def sanitize_filename_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower())
    return cleaned.strip("_.-") or "project"


def project_queries_filename(project_name: str, fallback_stem: str) -> str:
    project_name = (project_name or "").strip()
    if "/" in project_name:
        parts = [sanitize_filename_segment(part) for part in project_name.split("/") if part.strip()]
        if parts:
            return f"{'__'.join(parts)}__queries.json"
    return f"{sanitize_filename_segment(project_name or fallback_stem)}__queries.json"


def load_benchmark(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Benchmark file content must be a JSON object.")
    issues = data.get("issues")
    if not isinstance(issues, list):
        raise ValueError("Benchmark file must contain an 'issues' array.")
    return data


def export_queries(
    data: Dict[str, Any],
    issue_limit: int | None,
    include_empty_java: bool,
    extra_prompt: str,
) -> List[Dict[str, Any]]:
    issues = data.get("issues", [])
    if issue_limit:
        issues = issues[:issue_limit]

    entries: List[Dict[str, Any]] = []
    for issue in issues:
        expected_index = build_expected_index(issue)
        expected_paths = sorted(target.file_path for target in expected_index.targets.values())
        expected_count = len(expected_paths)
        if expected_count == 0 and not include_empty_java:
            continue
        issue_number = issue.get("number")
        title = str(issue.get("title", "") or "")
        description = issue.get("description_message")
        query = build_query_text(title, description, extra_prompt)
        entries.append(
            {
                "issue_number": issue_number,
                "query": query,
                "expected_classes_paths": expected_paths,
            }
        )
    return entries


def main() -> int:
    args = parse_args()
    benchmark_path = Path(args.benchmark_file).expanduser().resolve()
    if not benchmark_path.is_file():
        raise SystemExit(f"[ERROR] Benchmark file not found: {benchmark_path}")

    benchmark = load_benchmark(benchmark_path)
    entries = export_queries(
        data=benchmark,
        issue_limit=args.issue_limit,
        include_empty_java=args.include_empty_java,
        extra_prompt=args.extra_prompt,
    )

    if args.output_file:
        output_path = Path(args.output_file).expanduser().resolve()
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
        filename = project_queries_filename(
            project_name=str(benchmark.get("project_name", "") or ""),
            fallback_stem=benchmark_path.stem,
        )
        output_path = output_dir / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(entries, file, indent=2, ensure_ascii=False)

    print(f"Queries written to: {output_path}")
    print(f"Total queries: {len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
