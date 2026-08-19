#!/usr/bin/env python3
"""Convert an instrumentation trace (.jsonl) into a benchmark dataset file.

Each trace line is produced by running the benchmark on a specific dataset.
This script matches dataset issues with trace lines through the issue title
(dataset ``issues[].title`` == trace ``issue_title``) and produces a copy of
the dataset where each issue's ``description_message`` is replaced by the
concatenation of the ``system`` and ``user`` messages found in the trace's
``final_message`` field.

Issues without a corresponding trace line are dropped from the output dataset.
"""

import argparse
import json
import sys


def build_trace_index(trace_path):
    """Return a mapping ``issue_title -> concatenated final_message content``.

    The content is the concatenation of the ``system`` message content followed
    by the ``user`` message content found in ``final_message``.
    """
    index = {}
    with open(trace_path, encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as error:
                print(
                    f"[warn] skipping malformed JSON on line {line_number}: {error}",
                    file=sys.stderr,
                )
                continue

            title = entry.get("issue_title")
            if not title:
                print(
                    f"[warn] skipping line {line_number}: missing 'issue_title'",
                    file=sys.stderr,
                )
                continue

            final_message = entry.get("final_message")
            if not isinstance(final_message, list):
                print(
                    f"[warn] skipping line {line_number}: 'final_message' is not a list",
                    file=sys.stderr,
                )
                continue

            content = concatenate_messages(final_message)
            if content is None:
                print(
                    f"[warn] skipping line {line_number}: could not extract message content",
                    file=sys.stderr,
                )
                continue

            index[title] = content
    return index


def concatenate_messages(final_message):
    """Concatenate the ``system`` then ``user`` message contents.

    Falls back to concatenating every message content in order when the roles
    are not the expected ones. Returns ``None`` when no content is available.
    """
    by_role = {}
    for message in final_message:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role and content is not None:
            by_role[role] = content

    parts = []
    if "system" in by_role:
        parts.append(by_role["system"])
    if "user" in by_role:
        parts.append(by_role["user"])

    if not parts:
        parts = [
            message.get("content", "")
            for message in final_message
            if isinstance(message, dict) and message.get("content") is not None
        ]

    if not parts:
        return None
    return "\n\n".join(parts)


def convert(dataset, trace_index):
    """Return a new dataset dict with matched, replaced issues only."""
    result = dict(dataset)
    issues = dataset.get("issues", [])

    new_issues = []
    matched_titles = set()
    for issue in issues:
        title = issue.get("title")
        if title in trace_index:
            new_issue = dict(issue)
            new_issue["description_message"] = trace_index[title]
            new_issues.append(new_issue)
            matched_titles.add(title)

    result["issues"] = new_issues
    result["total_relevant_issues"] = len(new_issues)
    return result, matched_titles


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Convert an instrumentation trace (.jsonl) into a benchmark dataset "
            "file, replacing each issue's description_message with the trace's "
            "final_message content."
        )
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="Path to the source dataset JSON file used for the benchmark run.",
    )
    parser.add_argument(
        "-t",
        "--trace",
        required=True,
        help="Path to the instrumentation trace .jsonl file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path of the generated dataset JSON file.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    try:
        with open(args.dataset, encoding="utf-8") as handle:
            dataset = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        print(f"[error] cannot read dataset '{args.dataset}': {error}", file=sys.stderr)
        return 1

    try:
        trace_index = build_trace_index(args.trace)
    except OSError as error:
        print(f"[error] cannot read trace '{args.trace}': {error}", file=sys.stderr)
        return 1

    if not trace_index:
        print("[error] no usable trace entries found.", file=sys.stderr)
        return 1

    result, matched_titles = convert(dataset, trace_index)

    try:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as error:
        print(f"[error] cannot write output '{args.output}': {error}", file=sys.stderr)
        return 1

    total_issues = len(dataset.get("issues", []))
    matched = len(matched_titles)
    dropped = total_issues - matched
    unused = sorted(set(trace_index) - matched_titles)

    print(f"Dataset issues:        {total_issues}")
    print(f"Matched (kept):        {matched}")
    print(f"Dropped (no trace):    {dropped}")
    print(f"Unused trace entries:  {len(unused)}")
    for title in unused:
        print(f"  - {title}")
    print(f"Output written to:     {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
