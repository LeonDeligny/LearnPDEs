"""Reload the documentation hooks on each build, including during mkdocs serve.

MkDocs caches registered hook modules for the lifetime of its process. Keep
this adapter small and reload hooks.py in on_config so edits to asset paths
and Markdown handling take effect in an already-running preview server.
"""

from __future__ import annotations

from pathlib import Path
from runpy import run_path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

HOOK_PATH = Path(__file__).with_name('hooks.py')
_handlers = {}


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    global _handlers
    _handlers = run_path(str(HOOK_PATH))
    return config


def on_files(files: Files, config: MkDocsConfig) -> Files:
    return _handlers['on_files'](files, config)


def on_page_markdown(
    markdown: str, page: Page, config: MkDocsConfig, files: Files
) -> str:
    return _handlers['on_page_markdown'](markdown, page, config, files)
