"""Reuse the repository's tutorials and animations without duplicating sources."""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File, Files
from mkdocs.structure.pages import Page
from mkdocs.utils import get_relative_url

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ('exponential', 'cosinus', 'laplace', 'poiseuille')


def _load_cylinder_result(config: MkDocsConfig) -> MkDocsConfig:
    """Keep dashboard diagnostics tied to the displayed checkpoint's report."""
    plots = json.loads(
        (ROOT / 'assets/examples/cylinder/plots/diagnostics.json').read_text()
    )
    report_path = f'assets/verification/cylinder/{plots["run_id"]}.json'
    report = json.loads((ROOT / report_path).read_text())
    animation = json.loads(
        (ROOT / 'assets/examples/cylinder/plots/training-animation.json').read_text()
    )
    if plots['checkpoint_sha256'] != report['checkpoint_sha256']:
        raise PluginError(
            'Cylinder plots and verification must use the same checkpoint.'
        )
    if animation['checkpoint_sha256'] != report['checkpoint_sha256']:
        raise PluginError('Cylinder animation must end at the displayed checkpoint.')
    if report['reference_type'] != 'none' or report['project_acceptance']:
        raise PluginError(
            'Review the cylinder result presentation: its verification status changed.'
        )
    metrics = []
    for label, name in (
        ('Continuity RMS', 'continuity_rms'),
        ('Horizontal momentum RMS', 'momentum_u_rms'),
        ('Vertical momentum RMS', 'momentum_v_rms'),
        ('Outlet flux error', 'outlet_flow_relative_error'),
    ):
        value = max(e['metrics'][name] for e in report['evaluations'])
        display = (
            f'{100 * value:.5f}%'
            if name == 'outlet_flow_relative_error'
            else f'{value:.5f}'
        )
        metrics.append({'label': label, 'value': display})
    config.extra['cylinder_result'] = {
        'report': report_path,
        'metrics': metrics,
        'passed': report['single_run_checks_passed'],
        'seed': report['training_seed'],
        'points': f'{report["training_settings"]["training_samples"]:,}',
        'animation_frames': len(animation['frames']),
        'animation_steps': f'{animation["frames"][-1]["step"]:,}',
        'evaluation_counts': ' and '.join(
            f'{e["interior_count"]:,}' for e in report['evaluations']
        ),
    }
    return config


def _require_examples() -> None:
    required = [
        f'assets/examples/{scenario}/{name}'
        for scenario in SCENARIOS
        for name in ('training.gif', 'training.html', 'plotly.min.js', 'run.json')
    ]
    required += [
        f'assets/examples/cosinus_comparison/{name}'
        for name in ('comparison.html', 'plotly.min.js', 'run.json')
    ]
    for asset in required:
        if not (ROOT / asset).is_file():
            raise PluginError(
                f'Missing published example {asset}; see assets/README.md.'
            )


def _prepare_figure_asset(file: File, path: Path, category: str) -> None:
    if category == 'examples' and path.suffix == '.html':
        content = file.content_string
        if '<meta name="robots" content="noindex">' not in content:
            content = content.replace(
                '</head>', '<meta name="robots" content="noindex"></head>'
            )
        if path.name == 'training.html':
            # Style the embed shell without changing scientific figure fonts.
            stylesheet = get_relative_url('assets/site/figure.css', file.url)
            content = content.replace(
                '</head>',
                f'<link rel="stylesheet" href="{stylesheet}"></head>',
            )
        file.content_string = content


def on_files(files: Files, config: MkDocsConfig) -> Files:
    if (ROOT / 'docs/assets').exists():
        raise PluginError(
            'Keep shared assets under assets/; docs/assets/ would create a second source.'
        )
    _load_cylinder_result(config)
    _require_examples()
    # Copy the canonical published assets into the disposable site build.
    # Working runs and archived provenance never become website sources.
    for category in ('examples', 'verification', 'site'):
        directory = ROOT / 'assets' / category
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or any(
                part.startswith(('.', '_'))
                for part in path.relative_to(directory).parts
            ):
                continue
            asset = path.relative_to(ROOT).as_posix()
            file = File(
                asset,
                src_dir=str(ROOT),
                dest_dir=config.site_dir,
                use_directory_urls=config.use_directory_urls,
            )
            _prepare_figure_asset(file, path, category)
            files.append(file)
    return files


def on_page_markdown(
    markdown: str, page: Page, config: MkDocsConfig, files: Files
) -> str:
    if not page.meta.get('title') or not page.meta.get('description'):
        raise PluginError(f'{page.file.src_uri} needs a title and description')

    if page.file.src_uri.startswith('tutorials/') and page.file.name != 'index':
        required = (
            'scenario',
            'short_title',
            'topic',
            'equation',
        )
        missing = [key for key in required if not page.meta.get(key)]
        if missing:
            raise PluginError(
                f'{page.file.src_uri}: missing tutorial metadata {missing}'
            )

    def interactive_plot(match: re.Match[str]) -> str:
        label, name = match.groups()
        source = get_relative_url(f'assets/examples/{name}/training.html', page.url)
        fallback = get_relative_url(f'assets/examples/{name}/training.gif', page.url)
        height = {'cosinus': 1160, 'poiseuille': 1280}.get(name, 960)
        return (
            '<figure class="training-figure"><figcaption>Training figure</figcaption>'
            f'<iframe class="training-plot training-plot-{name}" src="{source}" '
            f'title="{escape(label, quote=True)} — interactive training" '
            f'loading="lazy" height="{height}"></iframe>\n\n'
            f'<p><a href="{source}">Open interactive figure</a> · '
            f'<a href="{fallback}">View GIF</a></p></figure>'
        )

    # GitHub keeps the GIFs; the website embeds the corresponding Plotly exports.
    markdown = re.sub(
        r'!\[([^\]]+)\]\(../../assets/examples/(exponential|cosinus|laplace|poiseuille)/training\.gif\)',
        interactive_plot,
        markdown,
    )

    comparison_source = get_relative_url(
        'assets/examples/cosinus_comparison/comparison.html', page.url
    )
    markdown = markdown.replace(
        '<!-- cosinus-comparison -->',
        '<figure class="training-figure"><figcaption>Compare all derivative orders</figcaption>'
        f'<iframe class="training-plot training-plot-comparison" src="{comparison_source}" '
        'title="Cosine solution comparison across derivative orders" loading="lazy" height="1500"></iframe>'
        f'<p><a href="{comparison_source}">Open the full comparison</a></p></figure>',
    )

    def repository_link(match: re.Match[str]) -> str:
        # Markdown stays readable on GitHub; adapt repository links for the site.
        target = match.group(1)
        source_path, separator, fragment = target.partition('#')
        if page.file.abs_src_path is None:
            raise PluginError('Repository links require a source file.')
        source = (Path(page.file.abs_src_path).parent / source_path).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file():
            raise PluginError(f'{page.file.src_uri}: missing repository file {target}')
        target = source.relative_to(ROOT).as_posix()
        suffix = separator + fragment
        if target.startswith('assets/'):
            return f']({get_relative_url(target, page.file.src_uri)}{suffix})'
        return f']({config.repo_url}/blob/main/{target}{suffix})'

    return re.sub(
        r'\]\(((?:\.\./)+(?:assets|learnpdes|tests)/[^\s)]+)\)',
        repository_link,
        markdown,
    )
