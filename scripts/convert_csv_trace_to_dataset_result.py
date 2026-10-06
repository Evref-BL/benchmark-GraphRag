#!/usr/bin/env python3
"""Convert issue responses from a CSV file into a benchmark dataset result.

The CSV must contain two columns: ``id`` (the issue number) and ``json`` (the
JSON response produced by the LLM).  Only issues present in both input files
are kept in the output, and the decoded CSV response is stored in each issue's
``response`` field.
"""

import argparse
import csv
import json
import sys


def build_response_index(csv_path):
    """Return a mapping from issue number to decoded LLM response."""
    index = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")

        required = {"id", "json"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV is missing required column(s): {', '.join(sorted(missing))}"
            )

        for line_number, row in enumerate(reader, start=2):
            raw_id = (row.get("id") or "").strip()
            raw_json = row.get("json")
            if not raw_id:
                print(
                    f"[warn] skipping CSV line {line_number}: missing 'id'",
                    file=sys.stderr,
                )
                continue
            if raw_json is None or not raw_json.strip():
                print(
                    f"[warn] skipping CSV line {line_number}: missing 'json'",
                    file=sys.stderr,
                )
                continue

            try:
                issue_id = int(raw_id)
            except ValueError:
                print(
                    f"[warn] skipping CSV line {line_number}: invalid issue id {raw_id!r}",
                    file=sys.stderr,
                )
                continue

            try:
                response = json.loads(raw_json)
            except json.JSONDecodeError as error:
                print(
                    f"[warn] skipping CSV line {line_number}: invalid JSON: {error}",
                    file=sys.stderr,
                )
                continue

            index[issue_id] = response
    return index


def convert(dataset, response_index):
    """Return a new dataset containing matched issues with their responses."""
    result = dict(dataset)
    new_issues = []
    matched_ids = set()

    for issue in dataset.get("issues", []):
        if not isinstance(issue, dict):
            continue
        issue_id = issue.get("number", issue.get("id"))
        try:
            issue_id = int(issue_id)
        except (TypeError, ValueError):
            continue

        if issue_id not in response_index:
            continue

        new_issue = dict(issue)
        new_issue["response"] = response_index[issue_id]
        new_issues.append(new_issue)
        matched_ids.add(issue_id)

    result["issues"] = new_issues
    result["total_relevant_issues"] = len(new_issues)
    return result, matched_ids


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Add LLM JSON responses from a CSV file to a benchmark dataset, "
            "matching responses with issues by issue id."
        )
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="Path to the source dataset JSON file.",
    )
    parser.add_argument(
        "-c",
        "--csv",
        required=True,
        dest="csv_path",
        help="Path to the CSV file containing the id,json columns.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path of the generated dataset result JSON file.",
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

    if not isinstance(dataset, dict) or not isinstance(dataset.get("issues"), list):
        print("[error] dataset must contain an 'issues' list.", file=sys.stderr)
        return 1

    try:
        response_index = build_response_index(args.csv_path)
    except (OSError, ValueError) as error:
        print(f"[error] cannot read CSV '{args.csv_path}': {error}", file=sys.stderr)
        return 1

    if not response_index:
        print("[error] no usable CSV responses found.", file=sys.stderr)
        return 1

    result, matched_ids = convert(dataset, response_index)

    try:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as error:
        print(f"[error] cannot write output '{args.output}': {error}", file=sys.stderr)
        return 1

    total_issues = len(dataset["issues"])
    unused = sorted(set(response_index) - matched_ids)
    print(f"Dataset issues:        {total_issues}")
    print(f"Matched (kept):        {len(matched_ids)}")
    print(f"Dropped (no response): {total_issues - len(matched_ids)}")
    print(f"Unused CSV entries:    {len(unused)}")
    for issue_id in unused:
        print(f"  - {issue_id}")
    print(f"Output written to:     {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
