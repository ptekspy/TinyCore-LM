from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.robotparser
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GITHUB_API = "https://api.github.com"
DEFAULT_ALLOWED_LICENSES = {
    "0BSD",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "ISC",
    "MIT",
    "MPL-2.0",
    "Unlicense",
}

INCLUDE_SUFFIXES = {
    ".ts",
    ".tsx",
    ".mts",
    ".cts",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".json",
    ".md",
}

SKIP_PATH_PARTS = {
    ".git",
    ".hg",
    ".next",
    ".nuxt",
    ".parcel-cache",
    ".turbo",
    "bower_components",
    "build",
    "coverage",
    "dist",
    "fixtures",
    "generated",
    "node_modules",
    "out",
    "storybook-static",
    "target",
    "vendor",
}

SKIP_FILENAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lockb",
    "npm-shrinkwrap.json",
}

DOC_LINK_KEYWORDS = {
    "api",
    "docs",
    "documentation",
    "getting-started",
    "guide",
    "install",
    "installation",
    "quickstart",
    "start",
    "usage",
}

DOC_PATH_HINTS = (
    "/docs",
    "/documentation",
    "/guide",
    "/guides",
    "/getting-started",
    "/quickstart",
    "/install",
    "/installation",
    "/usage",
    "/api",
)

DOC_SKIP_EXTENSIONS = {
    ".7z",
    ".avif",
    ".css",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".map",
    ".mp4",
    ".pdf",
    ".png",
    ".svg",
    ".tar",
    ".tgz",
    ".webm",
    ".woff",
    ".woff2",
    ".zip",
}

SECRET_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bASIA[0-9A-Z]{16}\b",
        r"\bghp_[A-Za-z0-9_]{30,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{40,}\b",
        r"\bsk-[A-Za-z0-9]{20,}\b",
        r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b",
        r"(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]",
    ]
]


@dataclass(frozen=True)
class RepoMeta:
    full_name: str
    clone_url: str
    default_branch: str
    html_url: str
    homepage_url: str
    stars: int
    forks: int
    size_kb: int
    license_spdx: str

    @property
    def score(self) -> int:
        return self.stars + self.forks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/training/typescript_github_top100_v0")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--candidate-pool", type=int, default=200)
    parser.add_argument("--max-repo-kb", type=int, default=250_000)
    parser.add_argument("--max-files-per-repo", type=int, default=1200)
    parser.add_argument("--max-file-bytes", type=int, default=200_000)
    parser.add_argument("--max-doc-pages-per-repo", type=int, default=40)
    parser.add_argument("--max-doc-bytes", type=int, default=300_000)
    parser.add_argument("--doc-timeout", type=float, default=20.0)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-doc-sites", action="store_true")
    parser.add_argument("--keep-zips", action="store_true")
    parser.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument("--allow-license", action="append", default=[])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = args.github_token or None
    allowed = sorted(DEFAULT_ALLOWED_LICENSES | set(args.allow_license))

    candidates = discover_candidates(args.candidate_pool, token=token, sleep=args.sleep)
    selected = select_repositories(candidates, args.top_n)
    (output_dir / "repo_candidates.json").write_text(json.dumps([repo.__dict__ for repo in candidates], indent=2) + "\n")
    (output_dir / "selected_repos.json").write_text(json.dumps([repo.__dict__ for repo in selected], indent=2) + "\n")

    if args.dry_run:
        manifest = manifest_base(args, selected, allowed, train_rows=0, val_rows=0, skipped=[])
        manifest["dry_run"] = True
        write_manifest(output_dir, manifest)
        print(json.dumps(manifest, indent=2))
        return

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    cache_dir = output_dir / "raw_zips"
    cache_dir.mkdir(parents=True, exist_ok=True)
    skipped: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    train_rows = 0
    val_rows = 0
    code_rows = 0
    doc_rows = 0
    docs_by_repo: dict[str, dict[str, object]] = {}

    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open("w", encoding="utf-8") as val_handle:
        for repo in selected:
            if repo.license_spdx not in allowed:
                skipped.append({"repo": repo.full_name, "reason": "license_not_allowed", "license": repo.license_spdx})
                continue
            if repo.size_kb > args.max_repo_kb:
                skipped.append({"repo": repo.full_name, "reason": "repo_too_large", "size_kb": repo.size_kb})
                continue

            try:
                zip_path = download_zipball(repo, cache_dir, token=token, sleep=args.sleep)
                repo_rows = 0
                repo_code_rows = 0
                for row in extract_rows(repo, zip_path, args.max_files_per_repo, args.max_file_bytes):
                    written_to_val = write_row(row, train_handle, val_handle, seen_hashes, args.val_fraction)
                    if written_to_val is None:
                        continue
                    val_rows += 1 if written_to_val else 0
                    train_rows += 0 if written_to_val else 1
                    repo_rows += 1
                    repo_code_rows += 1
                    code_rows += 1
                if not args.skip_doc_sites:
                    doc_seeds = find_doc_seed_urls(repo, zip_path)
                    repo_doc_rows = 0
                    doc_errors: list[str] = []
                    for row in crawl_doc_rows(
                        repo,
                        doc_seeds,
                        max_pages=args.max_doc_pages_per_repo,
                        max_bytes=args.max_doc_bytes,
                        timeout=args.doc_timeout,
                        sleep=args.sleep,
                    ):
                        if row.get("_error"):
                            doc_errors.append(str(row["_error"]))
                            continue
                        written_to_val = write_row(row, train_handle, val_handle, seen_hashes, args.val_fraction)
                        if written_to_val is None:
                            continue
                        val_rows += 1 if written_to_val else 0
                        train_rows += 0 if written_to_val else 1
                        repo_rows += 1
                        repo_doc_rows += 1
                        doc_rows += 1
                    docs_by_repo[repo.full_name] = {
                        "seed_urls": doc_seeds,
                        "rows": repo_doc_rows,
                        "errors": doc_errors[:5],
                    }
                if repo_rows == 0:
                    skipped.append({"repo": repo.full_name, "reason": "no_eligible_files"})
                if not args.keep_zips:
                    zip_path.unlink(missing_ok=True)
            except Exception as error:  # noqa: BLE001 - record and continue ingestion
                skipped.append({"repo": repo.full_name, "reason": "ingest_error", "error": str(error)})

    if not args.keep_zips:
        try:
            cache_dir.rmdir()
        except OSError:
            pass

    write_holdout(output_dir)
    manifest = manifest_base(args, selected, allowed, train_rows=train_rows, val_rows=val_rows, skipped=skipped)
    manifest["code_rows"] = code_rows
    manifest["doc_rows"] = doc_rows
    manifest["docs_by_repo"] = docs_by_repo
    manifest["sha256"] = {
        "train": sha256_file(train_path),
        "val": sha256_file(val_path),
        "eval_holdout": sha256_file(output_dir / "eval_holdout.jsonl"),
    }
    write_manifest(output_dir, manifest)
    print(json.dumps(manifest, indent=2))


def discover_candidates(candidate_pool: int, token: str | None, sleep: float) -> list[RepoMeta]:
    merged: dict[str, RepoMeta] = {}
    for sort in ("stars", "forks"):
        page = 1
        while len(merged) < candidate_pool * 2 and page <= max(1, (candidate_pool + 99) // 100):
            query = (
                f"{GITHUB_API}/search/repositories?q=language:TypeScript+archived:false+mirror:false"
                f"&sort={sort}&order=desc&per_page=100&page={page}"
            )
            payload = github_json(query, token=token)
            for item in payload.get("items", []):
                repo = repo_meta(item)
                merged.setdefault(repo.full_name, repo)
            page += 1
            time.sleep(sleep)
    return sorted(merged.values(), key=lambda repo: (-repo.score, -repo.stars, -repo.forks, repo.full_name))


def select_repositories(candidates: list[RepoMeta], top_n: int) -> list[RepoMeta]:
    return sorted(candidates, key=lambda repo: (-repo.score, -repo.stars, -repo.forks, repo.full_name))[:top_n]


def repo_meta(item: dict[str, object]) -> RepoMeta:
    license_info = item.get("license") if isinstance(item.get("license"), dict) else {}
    return RepoMeta(
        full_name=str(item["full_name"]),
        clone_url=str(item.get("clone_url", "")),
        default_branch=str(item.get("default_branch", "main")),
        html_url=str(item.get("html_url", "")),
        homepage_url=normalize_url(str(item.get("homepage") or "")),
        stars=int(item.get("stargazers_count", 0)),
        forks=int(item.get("forks_count", 0)),
        size_kb=int(item.get("size", 0)),
        license_spdx=str(license_info.get("spdx_id") or "NOASSERTION"),
    )


def download_zipball(repo: RepoMeta, cache_dir: Path, token: str | None, sleep: float) -> Path:
    safe_name = repo.full_name.replace("/", "__")
    zip_path = cache_dir / f"{safe_name}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path
    request = urllib.request.Request(
        f"{GITHUB_API}/repos/{repo.full_name}/zipball/{repo.default_branch}",
        headers=github_headers(token),
    )
    with urllib.request.urlopen(request, timeout=120) as response, zip_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    time.sleep(sleep)
    return zip_path


def extract_rows(repo: RepoMeta, zip_path: Path, max_files: int, max_file_bytes: int) -> Iterable[dict[str, str]]:
    emitted = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if emitted >= max_files:
                break
            if info.is_dir() or info.file_size == 0 or info.file_size > max_file_bytes:
                continue
            path = "/".join(Path(info.filename).parts[1:])
            if not is_allowed_path(path):
                continue
            raw = archive.read(info)
            if looks_binary(raw):
                continue
            text = raw.decode("utf-8", errors="ignore")
            if should_skip_text(text):
                continue
            row = make_row(repo, path, text)
            emitted += 1
            yield row


def write_row(row: dict[str, str], train_handle, val_handle, seen_hashes: set[str], val_fraction: float) -> bool | None:
    text_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
    if text_hash in seen_hashes:
        return None
    seen_hashes.add(text_hash)
    is_val = stable_fraction(row["id"]) < val_fraction
    handle = val_handle if is_val else train_handle
    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return is_val


def is_allowed_path(path: str) -> bool:
    lower = path.lower()
    parts = set(Path(lower).parts)
    name = Path(lower).name
    suffix = "".join(Path(lower).suffixes[-2:]) if lower.endswith(".d.ts") else Path(lower).suffix
    if parts & SKIP_PATH_PARTS:
        return False
    if name in SKIP_FILENAMES or name.endswith(".min.js") or name.endswith(".map"):
        return False
    if suffix == ".d.ts":
        return False
    return Path(lower).suffix in INCLUDE_SUFFIXES


def looks_binary(raw: bytes) -> bool:
    if b"\0" in raw:
        return True
    if not raw:
        return False
    sample = raw[:4096]
    control = sum(1 for byte in sample if byte < 9 or 13 < byte < 32)
    return control / len(sample) > 0.05


def should_skip_text(text: str) -> bool:
    if len(text.strip()) < 40:
        return True
    if any(len(line) > 1000 for line in text.splitlines()[:200]):
        return True
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def make_row(repo: RepoMeta, path: str, text: str) -> dict[str, str]:
    category = category_for_path(path)
    prompt = f"### Repository\n{repo.full_name}\n### File\n{path}\n### Content\n"
    response = clean_text(text) + "\n"
    row_id = hashlib.sha256(f"{repo.full_name}:{path}:{response}".encode("utf-8")).hexdigest()[:24]
    return {
        "id": f"github_ts_{row_id}",
        "split": "train_or_val",
        "category": category,
        "repo": repo.full_name,
        "repo_url": repo.html_url,
        "source_kind": "repo_file",
        "license": repo.license_spdx,
        "path": path,
        "prompt": prompt,
        "response": response,
        "text": prompt + response,
    }


def category_for_path(path: str) -> str:
    suffix = Path(path.lower()).suffix
    if suffix in {".ts", ".tsx", ".mts", ".cts"}:
        return "typescript"
    if suffix in {".js", ".jsx", ".mjs", ".cjs"}:
        return "javascript"
    if suffix == ".md":
        return "docs"
    if suffix == ".json":
        return "json"
    return "text"


def find_doc_seed_urls(repo: RepoMeta, zip_path: Path) -> list[str]:
    seeds: list[str] = []
    add_doc_seed(seeds, repo.homepage_url)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            path = "/".join(Path(info.filename).parts[1:])
            lower = path.lower()
            if info.is_dir() or info.file_size == 0 or info.file_size > 200_000:
                continue
            if Path(lower).name == "package.json":
                try:
                    payload = json.loads(archive.read(info).decode("utf-8", errors="ignore"))
                except json.JSONDecodeError:
                    continue
                for value in package_doc_urls(payload):
                    add_doc_seed(seeds, value)
            elif Path(lower).name.startswith("readme") and Path(lower).suffix in {".md", ".mdx", ".txt"}:
                text = archive.read(info).decode("utf-8", errors="ignore")
                for value in markdown_doc_urls(text):
                    add_doc_seed(seeds, value)
    return seeds[:5]


def add_doc_seed(seeds: list[str], url: str) -> None:
    normalized = normalize_url(url)
    if not normalized or not is_http_url(normalized):
        return
    host = urllib.parse.urlparse(normalized).netloc.lower()
    if host in {"github.com", "www.github.com"}:
        return
    if normalized not in seeds:
        seeds.append(normalized)


def package_doc_urls(payload: dict[str, object]) -> Iterable[str]:
    for key in ("homepage", "documentation", "docs"):
        value = payload.get(key)
        if isinstance(value, str):
            yield value
    bugs = payload.get("bugs")
    if isinstance(bugs, dict) and isinstance(bugs.get("url"), str):
        yield str(bugs["url"])


def markdown_doc_urls(text: str) -> Iterable[str]:
    for label, url in re.findall(r"\[([^\]]{1,120})\]\((https?://[^)\s]+)\)", text, flags=re.IGNORECASE):
        haystack = f"{label} {url}".lower()
        if any(keyword in haystack for keyword in DOC_LINK_KEYWORDS):
            yield url


def crawl_doc_rows(
    repo: RepoMeta,
    seeds: list[str],
    max_pages: int,
    max_bytes: int,
    timeout: float,
    sleep: float,
) -> Iterable[dict[str, str]]:
    emitted = 0
    seen: set[str] = set()
    robot_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
    for seed in seeds:
        if emitted >= max_pages:
            break
        root = doc_root(seed)
        queue = [seed]
        while queue and emitted < max_pages:
            url = normalize_url(queue.pop(0))
            if not url or url in seen or not is_http_url(url) or not is_same_doc_site(root, url):
                continue
            seen.add(url)
            if not is_allowed_doc_url(url) or not robot_allows(url, robot_cache, timeout):
                continue
            try:
                page = fetch_doc_page(url, max_bytes=max_bytes, timeout=timeout)
            except Exception as error:  # noqa: BLE001 - keep crawling other URLs
                yield {"_error": f"{url}: {error}"}
                continue
            if should_skip_text(page.text):
                continue
            row = make_doc_row(repo, url, page.title, page.text)
            emitted += 1
            yield row
            for link in page.links:
                absolute = normalize_url(urllib.parse.urljoin(url, link))
                if absolute and absolute not in seen and is_same_doc_site(root, absolute) and is_docish_url(absolute):
                    queue.append(absolute)
            time.sleep(sleep)


@dataclass(frozen=True)
class DocPage:
    title: str
    text: str
    links: list[str]


def fetch_doc_page(url: str, max_bytes: int, timeout: float) -> DocPage:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html, text/markdown;q=0.9, text/plain;q=0.8",
            "User-Agent": "TinyCore-LM-doc-ingestor",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "").lower()
        if not any(kind in content_type for kind in ("text/html", "text/markdown", "text/plain", "application/xhtml")):
            raise RuntimeError(f"unsupported content-type {content_type}")
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise RuntimeError("document_too_large")
    text = raw.decode("utf-8", errors="ignore")
    if "html" in content_type or "<html" in text[:500].lower():
        parser = ReadableHtmlParser()
        parser.feed(text)
        return DocPage(parser.title.strip(), clean_doc_text(parser.readable_text()), parser.links)
    return DocPage("", clean_doc_text(text), [])


class ReadableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.links: list[str] = []
        self._chunks: list[str] = []
        self._tag_stack: list[str] = []
        self._capture_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "title":
            self._capture_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(html.unescape(href))
        if tag in {"br", "p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "pre", "code"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._capture_title = False
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if self._tag_stack:
            self._tag_stack.pop()
        if tag in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "pre", "code"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._capture_title:
            self.title += data
        stripped = data.strip()
        if stripped:
            self._chunks.append(stripped)
            self._chunks.append(" ")

    def readable_text(self) -> str:
        return "".join(self._chunks)


def clean_doc_text(text: str) -> str:
    text = html.unescape(text)
    text = clean_text(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_doc_row(repo: RepoMeta, url: str, title: str, text: str) -> dict[str, str]:
    header = f"### Repository\n{repo.full_name}\n### Docs URL\n{url}\n"
    if title:
        header += f"### Page Title\n{title}\n"
    header += "### Content\n"
    response = clean_doc_text(text) + "\n"
    row_id = hashlib.sha256(f"{repo.full_name}:docs:{url}:{response}".encode("utf-8")).hexdigest()[:24]
    return {
        "id": f"github_ts_docs_{row_id}",
        "split": "train_or_val",
        "category": "docs_site",
        "repo": repo.full_name,
        "repo_url": repo.html_url,
        "docs_url": url,
        "source_kind": "docs_site",
        "license": repo.license_spdx,
        "path": url,
        "prompt": header,
        "response": response,
        "text": header + response,
    }


def normalize_url(url: str) -> str:
    url = html.unescape(url.strip().strip(").,;'\""))
    if not url:
        return ""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        return ""
    if not parsed.scheme:
        url = "https://" + url
        parsed = urllib.parse.urlparse(url)
    parsed = parsed._replace(fragment="")
    path = parsed.path or "/"
    normalized = parsed._replace(path=path, query=parsed.query).geturl()
    return normalized.rstrip("/") if path != "/" else normalized


def is_http_url(url: str) -> bool:
    return urllib.parse.urlparse(url).scheme in {"http", "https"}


def doc_root(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return parsed._replace(path="", params="", query="", fragment="").geturl()


def is_same_doc_site(root: str, url: str) -> bool:
    return urllib.parse.urlparse(root).netloc.lower() == urllib.parse.urlparse(url).netloc.lower()


def is_allowed_doc_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path.lower()).suffix
    if suffix in DOC_SKIP_EXTENSIONS:
        return False
    return parsed.scheme in {"http", "https"}


def is_docish_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    if path in {"", "/"}:
        return True
    return any(hint in path for hint in DOC_PATH_HINTS)


def robot_allows(url: str, cache: dict[str, urllib.robotparser.RobotFileParser], timeout: float) -> bool:
    parsed = urllib.parse.urlparse(url)
    root = parsed._replace(path="", params="", query="", fragment="").geturl()
    parser = cache.get(root)
    if parser is None:
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(urllib.parse.urljoin(root + "/", "robots.txt"))
        try:
            parser.read()
        except Exception:  # noqa: BLE001 - absent/unreadable robots means no explicit block
            parser = urllib.robotparser.RobotFileParser()
            parser.parse([])
        cache[root] = parser
    return parser.can_fetch("TinyCore-LM-doc-ingestor", url)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def stable_fraction(value: str) -> float:
    number = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)
    return number / float(0xFFFFFFFFFFFF)


def github_json(url: str, token: str | None) -> dict[str, object]:
    request = urllib.request.Request(url, headers=github_headers(token))
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"GitHub API request failed {error.code}: {detail}") from error


def github_headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "TinyCore-LM-dataset-ingestor",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def write_holdout(output_dir: Path) -> None:
    rows = [
        {
            "id": "github_ts_holdout_repo_summary",
            "split": "eval_holdout",
            "category": "typescript",
            "prompt": "Q:repo summary|A:",
            "response": "Summarize modules, entry points, tests, build scripts, and risky generated files before editing.\n",
            "text": "Q:repo summary|A:Summarize modules, entry points, tests, build scripts, and risky generated files before editing.\n",
        },
        {
            "id": "github_ts_holdout_patch",
            "split": "eval_holdout",
            "category": "typescript",
            "prompt": "Q:typescript patch rule|A:",
            "response": "Prefer small typed changes, run the relevant test command, and avoid editing generated bundles.\n",
            "text": "Q:typescript patch rule|A:Prefer small typed changes, run the relevant test command, and avoid editing generated bundles.\n",
        },
    ]
    with (output_dir / "eval_holdout.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def manifest_base(
    args: argparse.Namespace,
    selected: list[RepoMeta],
    allowed: list[str],
    train_rows: int,
    val_rows: int,
    skipped: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "name": "typescript_github_top100_v0",
        "source": "GitHub REST API repository search, public repository zipballs, and discovered public docs websites",
        "license_notes": "Default ingestion includes only allowlisted permissive licenses and skips unknown/non-permissive licenses.",
        "ranking": "merge top TypeScript repositories sorted by stars and forks, then rank by stars+forks",
        "top_n": args.top_n,
        "candidate_pool": args.candidate_pool,
        "allowed_licenses": allowed,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "num_documents": train_rows + val_rows,
        "selected_repositories": [repo.__dict__ for repo in selected],
        "skipped": skipped,
        "filters": {
            "max_repo_kb": args.max_repo_kb,
            "max_files_per_repo": args.max_files_per_repo,
            "max_file_bytes": args.max_file_bytes,
            "skip_doc_sites": args.skip_doc_sites,
            "max_doc_pages_per_repo": args.max_doc_pages_per_repo,
            "max_doc_bytes": args.max_doc_bytes,
            "include_suffixes": sorted(INCLUDE_SUFFIXES),
            "skip_path_parts": sorted(SKIP_PATH_PARTS),
            "secret_patterns": len(SECRET_PATTERNS),
        },
        "format": "jsonl with id, split, category, repo, repo_url, source_kind, license, path, prompt, response, text",
    }


def write_manifest(output_dir: Path, manifest: dict[str, object]) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
