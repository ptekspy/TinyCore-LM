from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "ingest_github_typescript_repos.py"
spec = importlib.util.spec_from_file_location("ingest_github_typescript_repos", SCRIPT)
assert spec is not None and spec.loader is not None
ingest = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ingest
spec.loader.exec_module(ingest)


def test_select_repositories_ranks_by_stars_plus_forks() -> None:
    repos = [
        ingest.RepoMeta("b/repo", "", "main", "", "", stars=50, forks=60, size_kb=1, license_spdx="MIT"),
        ingest.RepoMeta("a/repo", "", "main", "", "", stars=100, forks=1, size_kb=1, license_spdx="MIT"),
        ingest.RepoMeta("c/repo", "", "main", "", "", stars=20, forks=200, size_kb=1, license_spdx="MIT"),
    ]

    assert [repo.full_name for repo in ingest.select_repositories(repos, 2)] == ["c/repo", "b/repo"]


def test_ingest_path_filter_skips_generated_and_lock_files() -> None:
    assert ingest.is_allowed_path("src/index.ts") is True
    assert ingest.is_allowed_path("packages/app/src/view.tsx") is True
    assert ingest.is_allowed_path("node_modules/pkg/index.ts") is False
    assert ingest.is_allowed_path("dist/bundle.js") is False
    assert ingest.is_allowed_path("package-lock.json") is False
    assert ingest.is_allowed_path("src/types.d.ts") is False


def test_secret_filter_rejects_tokens_and_private_keys() -> None:
    assert ingest.should_skip_text("export const value = 1;\n") is True
    assert ingest.should_skip_text("const token = 'ghp_abcdefghijklmnopqrstuvwxyzABCDE';") is True
    assert ingest.should_skip_text("-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----") is True
    assert ingest.should_skip_text("export function add(a: number, b: number) {\n  return a + b;\n}\n") is False


def test_make_row_wraps_repository_metadata() -> None:
    repo = ingest.RepoMeta("owner/repo", "", "main", "https://github.com/owner/repo", "", 1, 2, 3, "MIT")
    row = ingest.make_row(repo, "src/index.ts", "export const value = 1;")

    assert row["repo"] == "owner/repo"
    assert row["license"] == "MIT"
    assert row["source_kind"] == "repo_file"
    assert "### Repository\nowner/repo" in row["text"]
    assert "### File\nsrc/index.ts" in row["text"]


def test_markdown_doc_urls_prefers_install_and_docs_links() -> None:
    text = """
    [Documentation](https://example.com/docs)
    [Install guide](https://example.com/guide/install)
    [Chat](https://discord.example.com/project)
    """

    assert list(ingest.markdown_doc_urls(text)) == [
        "https://example.com/docs",
        "https://example.com/guide/install",
    ]


def test_doc_url_helpers_stay_on_same_doc_site() -> None:
    assert ingest.normalize_url("example.com/docs#install") == "https://example.com/docs"
    assert ingest.is_docish_url("https://example.com/docs/install") is True
    assert ingest.is_docish_url("https://example.com/blog/company-news") is False
    assert ingest.is_same_doc_site("https://example.com", "https://example.com/docs") is True
    assert ingest.is_same_doc_site("https://example.com", "https://cdn.example.com/docs") is False


def test_html_parser_extracts_readable_text_and_links() -> None:
    parser = ingest.ReadableHtmlParser()
    parser.feed(
        """
        <html><head><title>Install Tiny</title><script>ignore()</script></head>
        <body><h1>Install</h1><p>npm install tiny</p><a href="/docs/usage">Usage</a></body></html>
        """
    )

    assert parser.title.strip() == "Install Tiny"
    assert "npm install tiny" in ingest.clean_doc_text(parser.readable_text())
    assert parser.links == ["/docs/usage"]


def test_make_doc_row_wraps_docs_metadata() -> None:
    repo = ingest.RepoMeta("owner/repo", "", "main", "https://github.com/owner/repo", "https://example.com", 1, 2, 3, "MIT")
    row = ingest.make_doc_row(repo, "https://example.com/docs/install", "Install", "npm install repo")

    assert row["category"] == "docs_site"
    assert row["source_kind"] == "docs_site"
    assert row["docs_url"] == "https://example.com/docs/install"
    assert "### Docs URL\nhttps://example.com/docs/install" in row["text"]
    assert "npm install repo" in row["text"]
