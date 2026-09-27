"""The website and repository Markdown consume the same published assets."""

import tempfile
import unittest
from pathlib import Path
from runpy import run_path
from unittest.mock import patch

from mkdocs.config import load_config
from mkdocs.exceptions import PluginError
from mkdocs.structure.files import Files

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'docs/website/mkdocs.yml'


class TestSharedAssets(unittest.TestCase):
    def test_site_assets_come_from_the_canonical_repository_tree(self) -> None:
        config = load_config(config_file=str(CONFIG))
        config = config.plugins.on_config(config)
        files = config.plugins.on_files(Files([]), config=config)
        assets = {file.src_uri: file for file in files}
        for name in (
            'assets/examples/cylinder/plots/flow-fields.png',
            'assets/examples/cylinder/plots/diagnostics.json',
            'assets/verification/index.json',
            'assets/site/extra.css',
            'assets/site/fonts/Manrope.ttf',
        ):
            with self.subTest(asset=name):
                source = assets[name].abs_src_path
                assert source is not None
                self.assertEqual(Path(source), ROOT / name)
        figure = assets['assets/examples/laplace/training.html']
        self.assertIn('href="../../site/figure.css"', figure.content_string)
        self.assertFalse(
            any(
                name.startswith(
                    (
                        'assets/runs/',
                        'assets/comparisons/',
                        'assets/examples/_provenance/',
                    )
                )
                for name in assets
            )
        )

    def test_build_rejects_a_second_asset_tree_in_docs(self) -> None:
        on_files = run_path(str(ROOT / 'docs/website/hooks.py'))['on_files']
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'docs/assets').mkdir(parents=True)
            with patch.dict(on_files.__globals__, ROOT=root):
                with self.assertRaisesRegex(PluginError, 'second source'):
                    on_files(Files([]), load_config(config_file=str(CONFIG)))


if __name__ == '__main__':
    unittest.main()
