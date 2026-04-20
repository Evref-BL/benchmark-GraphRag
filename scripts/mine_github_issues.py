#!/usr/bin/env python3
"""Mine GitHub issues linked to merged pull requests for benchmark generation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, Optional, Set, Tuple

PRRef = Tuple[str, str, int]

GITHUB_REPO_HOSTS = {"github.com", "www.github.com"}
PR_URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/pull/(?P<number>\d+)",
    re.IGNORECASE,
)
API_PULL_URL_RE = re.compile(
    r"https?://api\.github\.com/repos/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/pulls/(?P<number>\d+)",
    re.IGNORECASE,
)
OWNER_REPO_REF_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)#(?P<number>\d+)"
)
REPO_SLUG_RE = re.compile(r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)$")
API_REPO_URL_RE = re.compile(
    r"^https?://api\.github\.com/repos/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)$",
    re.IGNORECASE,
)
DEFAULT_ENV_FILENAME = ".env.github"
DEFAULT_OUTPUT_DIRNAME = "mined_projects"
PLACEHOLDER_TOKENS = {
    "YOUR_GITHUB_TOKEN_HERE",
    "YOUR_TOKEN_HERE",
    "GITHUB_TOKEN_HERE",
}


class GitHubAPIError(RuntimeError):
    """Raised when GitHub API requests fail."""

    def __init__(self, status_code: int, url: str, message: str) -> None:
        super().__init__(f"GitHub API error {status_code} for {url}: {message}")
        self.status_code = status_code
        self.url = url
        self.message = message


def normalize_pr_ref(owner: str, repo: str, number: int) -> PRRef:
    return owner.lower(), repo.lower(), int(number)


def parse_iso_datetime(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_repo_from_input(repo_input: str) -> Tuple[str, str]:
    repo_input = repo_input.strip()
    match = REPO_SLUG_RE.match(repo_input)
    if match:
        return match.group("owner"), match.group("repo")

    parsed = urllib.parse.urlparse(repo_input)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in GITHUB_REPO_HOSTS:
        raise ValueError(f"Not a valid GitHub repository URL or slug: {repo_input}")

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        raise ValueError(f"Repository path is incomplete: {repo_input}")

    owner = path_parts[0]
    repo = path_parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def parse_repo_from_api_url(repo_url: str) -> Optional[Tuple[str, str]]:
    if not repo_url:
        return None
    match = API_REPO_URL_RE.match(repo_url)
    if not match:
        return None
    return match.group("owner"), match.group("repo")


def extract_pr_refs_from_text(text: str) -> Set[PRRef]:
    refs: Set[PRRef] = set()
    if not text:
        return refs

    for regex in (PR_URL_RE, API_PULL_URL_RE, OWNER_REPO_REF_RE):
        for match in regex.finditer(text):
            refs.add(
                normalize_pr_ref(
                    owner=match.group("owner"),
                    repo=match.group("repo"),
                    number=int(match.group("number")),
                )
            )
    return refs


def extract_pr_refs_from_object(
    obj: Any, default_owner: Optional[str] = None, default_repo: Optional[str] = None
) -> Set[PRRef]:
    refs: Set[PRRef] = set()
    if obj is None:
        return refs

    if isinstance(obj, str):
        return extract_pr_refs_from_text(obj)

    if isinstance(obj, list):
        for item in obj:
            refs.update(extract_pr_refs_from_object(item, default_owner, default_repo))
        return refs

    if not isinstance(obj, dict):
        return refs

    number = obj.get("number")
    if obj.get("pull_request") and isinstance(number, int):
        owner_repo = parse_repo_from_api_url(obj.get("repository_url", ""))
        if owner_repo:
            refs.add(normalize_pr_ref(owner_repo[0], owner_repo[1], number))
        elif default_owner and default_repo:
            refs.add(normalize_pr_ref(default_owner, default_repo, number))

    html_url = obj.get("html_url")
    if isinstance(html_url, str):
        refs.update(extract_pr_refs_from_text(html_url))

    url = obj.get("url")
    if isinstance(url, str):
        refs.update(extract_pr_refs_from_text(url))

    for value in obj.values():
        refs.update(extract_pr_refs_from_object(value, default_owner, default_repo))
    return refs


def parse_next_link(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None

    for part in link_header.split(","):
        section = part.strip()
        if not section:
            continue
        segments = [segment.strip() for segment in section.split(";")]
        if len(segments) < 2:
            continue
        if 'rel="next"' not in segments[1:]:
            continue
        url_fragment = segments[0]
        if url_fragment.startswith("<") and url_fragment.endswith(">"):
            return url_fragment[1:-1]
    return None


class GitHubClient:
    API_BASE = "https://api.github.com"

    def __init__(self, token: Optional[str]) -> None:
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "graph-rag-issue-miner",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def _build_url(self, path_or_url: str, params: Optional[Dict[str, Any]] = None) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            parsed = urllib.parse.urlparse(path_or_url)
            query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
            if params:
                for key, value in params.items():
                    if value is None:
                        continue
                    query[key] = [str(value)]
            new_query = urllib.parse.urlencode(query, doseq=True)
            return parsed._replace(query=new_query).geturl()

        base = urllib.parse.urljoin(self.API_BASE + "/", path_or_url.lstrip("/"))
        if not params:
            return base
        cleaned = {k: str(v) for k, v in params.items() if v is not None}
        return f"{base}?{urllib.parse.urlencode(cleaned)}"

    def _request(self, url: str) -> Tuple[Any, Dict[str, str]]:
        req = urllib.request.Request(url=url, headers=self.headers)
        try:
            with urllib.request.urlopen(req) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body) if body else None
                return data, dict(response.headers.items())
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            message = body
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict) and parsed.get("message"):
                    message = str(parsed.get("message"))
            except json.JSONDecodeError:
                pass

            remaining = error.headers.get("X-RateLimit-Remaining")
            reset = error.headers.get("X-RateLimit-Reset")
            if error.code == 403 and remaining == "0":
                if reset and reset.isdigit():
                    reset_at = dt.datetime.fromtimestamp(int(reset), tz=dt.timezone.utc)
                    message = f"{message}. Rate limit resets at {reset_at.isoformat()}."
                else:
                    message = f"{message}. GitHub API rate limit exceeded."

            raise GitHubAPIError(status_code=error.code, url=url, message=message) from error

    def get(self, path_or_url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self._build_url(path_or_url, params)
        data, _ = self._request(url)
        return data

    def paginate(self, path_or_url: str, params: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
        next_url = self._build_url(path_or_url, params)
        while next_url:
            data, headers = self._request(next_url)
            if not isinstance(data, list):
                raise GitHubAPIError(
                    status_code=500, url=next_url, message="Expected list response for pagination."
                )
            for item in data:
                yield item
            next_url = parse_next_link(headers.get("Link"))


def collect_issue_pr_refs(
    client: GitHubClient, owner: str, repo: str, issue: Dict[str, Any]
) -> Set[PRRef]:
    refs: Set[PRRef] = set()
    refs.update(extract_pr_refs_from_object(issue, owner, repo))

    comments_url = issue.get("comments_url")
    if comments_url:
        for comment in client.paginate(comments_url, {"per_page": 100}):
            refs.update(extract_pr_refs_from_object(comment, owner, repo))

    timeline_path = f"/repos/{owner}/{repo}/issues/{issue['number']}/timeline"
    for event in client.paginate(timeline_path, {"per_page": 100}):
        refs.update(extract_pr_refs_from_object(event, owner, repo))

    return refs


def fetch_pull_request(
    client: GitHubClient,
    owner: str,
    repo: str,
    number: int,
    cache: Dict[PRRef, Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    key = normalize_pr_ref(owner, repo, number)
    if key in cache:
        return cache[key]

    try:
        pr = client.get(f"/repos/{owner}/{repo}/pulls/{number}")
    except GitHubAPIError as error:
        if error.status_code == 404:
            cache[key] = None
            return None
        raise

    if not isinstance(pr, dict):
        cache[key] = None
        return None

    impacted_files = fetch_pull_request_files(client, owner, repo, number)
    result = {
        "number": pr.get("number"),
        "title": pr.get("title"),
        "url": pr.get("html_url"),
        "repository": f"{owner.lower()}/{repo.lower()}",
        "merged_at": pr.get("merged_at"),
        "impacted_files": impacted_files,
        "impacted_files_count": len(impacted_files),
    }
    cache[key] = result
    return result


def fetch_pull_request_files(client: GitHubClient, owner: str, repo: str, number: int) -> list[str]:
    files: list[str] = []
    seen: Set[str] = set()
    for item in client.paginate(f"/repos/{owner}/{repo}/pulls/{number}/files", {"per_page": 100}):
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        if not isinstance(filename, str) or filename in seen:
            continue
        seen.add(filename)
        files.append(filename)
    return files


def mine_repository(
    client: GitHubClient,
    owner: str,
    repo: str,
    max_issues: Optional[int] = None,
    progress_every: int = 25,
) -> Dict[str, Any]:
    repo_slug = f"{owner}/{repo}"
    print(f"[{repo_slug}] Mining closed issues...", file=sys.stderr)

    issue_entries = []
    scanned_closed_issues = 0
    pr_cache: Dict[PRRef, Optional[Dict[str, Any]]] = {}

    params = {"state": "closed", "sort": "updated", "direction": "desc", "per_page": 100}
    for issue in client.paginate(f"/repos/{owner}/{repo}/issues", params):
        if issue.get("pull_request"):
            continue

        scanned_closed_issues += 1
        if max_issues and scanned_closed_issues > max_issues:
            break
        if progress_every and scanned_closed_issues % progress_every == 0:
            print(f"[{repo_slug}] Scanned {scanned_closed_issues} closed issues...", file=sys.stderr)

        closed_at = parse_iso_datetime(issue.get("closed_at"))
        if not closed_at:
            continue

        pr_refs = collect_issue_pr_refs(client, owner, repo, issue)
        if not pr_refs:
            continue

        merged_prs = []
        for ref_owner, ref_repo, ref_number in sorted(pr_refs):
            pr_data = fetch_pull_request(client, ref_owner, ref_repo, ref_number, pr_cache)
            if not pr_data or not pr_data.get("merged_at"):
                continue
            merged_prs.append(pr_data)

        if not merged_prs:
            continue

        has_merge_before_close = any(
            (parse_iso_datetime(pr.get("merged_at")) or dt.datetime.max.replace(tzinfo=dt.timezone.utc))
            <= closed_at
            for pr in merged_prs
        )
        if not has_merge_before_close:
            continue

        merged_prs.sort(key=lambda item: (item.get("merged_at") or "", item.get("repository") or "", item.get("number") or 0))
        issue_entries.append(
            {
                "number": issue.get("number"),
                "title": issue.get("title"),
                "url": issue.get("html_url"),
                "description_message": issue.get("body"),
                "created_at": issue.get("created_at"),
                "closed_at": issue.get("closed_at"),
                "linked_merged_pull_request_urls": [pr["url"] for pr in merged_prs if pr.get("url")],
                "linked_merged_pull_requests": merged_prs,
            }
        )

    issue_entries.sort(key=lambda item: item.get("number") or 0)
    return {
        "project_name": repo_slug.lower(),
        "github_url": f"https://github.com/{owner}/{repo}",
        "mined_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "total_closed_issues_scanned": scanned_closed_issues,
        "total_relevant_issues": len(issue_entries),
        "issues": issue_entries,
    }


def make_output_filename(owner: str, repo: str) -> str:
    timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_owner = re.sub(r"[^A-Za-z0-9_.-]+", "_", owner.lower())
    safe_repo = re.sub(r"[^A-Za-z0-9_.-]+", "_", repo.lower())
    return f"{safe_owner}__{safe_repo}__{timestamp}.json"


def default_output_dir() -> str:
    """Return the default output directory anchored to project root."""
    project_root = Path(__file__).resolve().parent.parent
    return str(project_root / DEFAULT_OUTPUT_DIRNAME)


def parse_env_line_for_var(line: str, var_name: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None

    key, raw_value = stripped.split("=", 1)
    if key.strip() != var_name:
        return None
    value = raw_value.strip()
    if value and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value or None


def normalize_token_value(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    cleaned = token.strip()
    if not cleaned:
        return None
    if cleaned.upper() in PLACEHOLDER_TOKENS:
        return None
    return cleaned


def read_token_from_env_file(env_path: Path) -> Optional[str]:
    if not env_path.is_file():
        return None
    try:
        with env_path.open("r", encoding="utf-8") as file:
            for line in file:
                parsed = parse_env_line_for_var(line, "GITHUB_TOKEN")
                if parsed:
                    return parsed
    except OSError:
        return None
    return None


def resolve_github_token(cli_token: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    cli_value = normalize_token_value(cli_token)
    if cli_value:
        return cli_value, "--token"

    env_value = normalize_token_value(os.getenv("GITHUB_TOKEN"))
    if env_value:
        return env_value, "GITHUB_TOKEN"

    cwd_env = Path.cwd() / DEFAULT_ENV_FILENAME
    script_env = Path(__file__).resolve().parent.parent / DEFAULT_ENV_FILENAME
    candidates = []
    for candidate in (cwd_env, script_env):
        if candidate not in candidates:
            candidates.append(candidate)

    for path in candidates:
        loaded_token = read_token_from_env_file(path)
        normalized = normalize_token_value(loaded_token)
        if normalized:
            return normalized, str(path)
    return None, None


def parse_args() -> argparse.Namespace:
    def parse_issue_limit(value: Optional[str]) -> int:
        if value is None:
            return 200
        cleaned = str(value).strip()
        if cleaned == "":
            return 200
        try:
            parsed = int(cleaned)
        except ValueError as error:
            raise argparse.ArgumentTypeError("Issue limit must be an integer.") from error
        if parsed <= 0:
            raise argparse.ArgumentTypeError("Issue limit must be greater than 0.")
        return parsed

    parser = argparse.ArgumentParser(
        description=(
            "Mine GitHub issues linked to merged pull requests and export one benchmark JSON per repository."
        )
    )
    parser.add_argument(
        "repos",
        nargs="+",
        help="GitHub repository URL(s) or owner/repo slug(s).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=default_output_dir(),
        help=(
            "Directory where project mining JSON files will be written. "
            "Default: project-root/mined_projects."
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "GitHub token. Resolution order: --token, env GITHUB_TOKEN, then .env.github "
            "(cwd or project root). Strongly recommended to avoid rate limits."
        ),
    )
    parser.add_argument(
        "--issue-limit",
        "--max-issues",
        dest="max_issues",
        nargs="?",
        default=200,
        const=200,
        type=parse_issue_limit,
        help=(
            "Cap on closed issues scanned per repository. "
            "Defaults to 200 when omitted or empty."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N scanned closed issues.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token, token_source = resolve_github_token(args.token)
    if token_source and token_source.endswith(DEFAULT_ENV_FILENAME):
        print(f"Loaded GitHub token from {token_source}", file=sys.stderr)

    if not token:
        print(
            "Warning: no GitHub token configured. Unauthenticated requests are limited to 60 requests/hour.",
            file=sys.stderr,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    client = GitHubClient(token=token)
    failures = 0

    for raw_repo in args.repos:
        try:
            owner, repo = parse_repo_from_input(raw_repo)
            mined = mine_repository(
                client=client,
                owner=owner,
                repo=repo,
                max_issues=args.max_issues,
                progress_every=args.progress_every,
            )
            output_path = os.path.join(args.output_dir, make_output_filename(owner, repo))
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(mined, file, indent=2, ensure_ascii=False)
            print(f"[{owner}/{repo}] Wrote output to {output_path}", file=sys.stderr)
        except (ValueError, GitHubAPIError) as error:
            failures += 1
            print(f"[ERROR] {raw_repo}: {error}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
