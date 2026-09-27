"""Check repeated config loads in the same process, as mkdocs serve rebuilds."""

import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import Mock, patch

from mkdocs.config import load_config
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

CONFIG_PATH = Path(__file__).with_name('mkdocs.yml')


class TestLiveHooks(unittest.TestCase):
    def test_cached_adapter_reloads_edited_asset_and_markdown_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / 'hooks.py'
            first = load_config(config_file=str(CONFIG_PATH))
            adapter = cast(dict[str, ModuleType], first.hooks)['live_hooks.py']
            with patch.object(adapter, 'HOOK_PATH', helper):
                for value in ('old-path', 'new-path'):
                    # Equal-length edits also catch timestamp-based bytecode caching.
                    helper.write_text(
                        f'def on_files(files, config):\n    return ["{value}"]\n'
                        'def on_page_markdown(markdown, page, config, files):\n'
                        f'    return markdown + " {value}"\n'
                    )
                    config = load_config(config_file=str(CONFIG_PATH))
                    self.assertIs(
                        cast(dict[str, ModuleType], config.hooks)['live_hooks.py'],
                        adapter,
                    )
                    config = config.plugins.on_config(config)
                    self.assertEqual(
                        config.plugins.on_files(Files([]), config=config), [value]
                    )
                    self.assertEqual(
                        config.plugins.on_page_markdown(
                            'content',
                            page=Mock(spec=Page),
                            config=config,
                            files=Files([]),
                        ),
                        f'content {value}',
                    )


if __name__ == '__main__':
    unittest.main()
