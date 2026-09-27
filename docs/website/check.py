"""Check the actual Pages artifact, including links under the repository prefix."""

import os
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree

from mkdocs.config import load_config


class Page(HTMLParser):
    def __init__(self, path: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.meta: dict[str, list[str]] = {}
        self.canonicals: list[str | None] = []
        self.title = ''
        self.in_title = False
        self.in_header = False
        self.header_repository_links: list[str | None] = []
        self.footers = 0
        self.sidebar_modes: list[str | None] = []
        self.header_menu_buttons = 0
        self.headings = 0
        self.math = 0
        self.scenarios: list[str] = []
        self.feed(path.read_text(encoding='utf-8'))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value or '' for name, value in attrs}
        self._record_metadata(tag, attributes)
        self._record_navigation(tag, attributes)
        self._record_content(tag, attributes)

    def _record_metadata(self, tag: str, attrs: dict[str, str]) -> None:
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'meta':
            name = attrs.get('name') or attrs.get('property', '')
            self.meta.setdefault(name, []).append(attrs.get('content', ''))
        if tag == 'link' and attrs.get('rel') == 'canonical':
            self.canonicals.append(attrs.get('href'))

    def _record_navigation(self, tag: str, attrs: dict[str, str]) -> None:
        if tag == 'title':
            self.in_title = True
        if tag == 'header':
            self.in_header = True
        if tag == 'footer':
            self.footers += 1
        if tag == 'input' and attrs.get('name') == 'sidebar-mode':
            self.sidebar_modes.append(attrs.get('value'))
        if tag == 'button' and 'menu-toggle' in attrs.get('class', '').split():
            self.header_menu_buttons += 1
        if (
            tag == 'a'
            and self.in_header
            and 'github-link' in attrs.get('class', '').split()
        ):
            self.header_repository_links.append(attrs.get('href'))

    def _record_content(self, tag: str, attrs: dict[str, str]) -> None:
        if tag == 'h1':
            self.headings += 1
        if tag == 'article' and 'data-scenario' in attrs:
            self.scenarios.append(attrs['data-scenario'])
        if 'arithmatex' in attrs.get('class', '').split():
            self.math += 1
        for attribute in ('href', 'src'):
            if attrs.get(attribute):
                self.links.append(attrs[attribute])

    def handle_endtag(self, tag: str) -> None:
        if tag == 'title':
            self.in_title = False
        if tag == 'header':
            self.in_header = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data


class SiteChecker:
    """Validate metadata, navigation and assets against one built site."""

    def __init__(self) -> None:
        self.config = load_config(
            config_file=str(Path(__file__).with_name('mkdocs.yml'))
        )
        self.site = Path(self.config.site_dir)
        base_url = self.config.site_url
        if base_url is None:
            raise ValueError('The documentation requires a site_url.')
        self.base_url: str = base_url
        self.base = urlsplit(self.base_url)
        self.errors: list[str] = []
        self.pages = {path: Page(path) for path in self.site.rglob('*.html')}
        self.interactive = {
            path: page
            for path, page in self.pages.items()
            if path.is_relative_to(self.site / 'assets' / 'examples')
        }
        self.content = {
            path: page
            for path, page in self.pages.items()
            if path.name != '404.html' and path not in self.interactive
        }
        if not self.content:
            raise SystemExit(
                'No built pages found. Run mkdocs build --strict -f docs/website/mkdocs.yml first.'
            )

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def page_url(self, path: Path) -> str:
        return urljoin(
            self.base_url,
            path.relative_to(self.site).as_posix().removesuffix('index.html'),
        )

    def _check_repository_assets(self) -> None:
        """Check the asset links GitHub renders without the website hooks."""
        root = Path(__file__).resolve().parents[2]
        for markdown in (root / 'README.md', *(root / 'docs').rglob('*.md')):
            for link in re.findall(r'\]\(([^\s)]+)\)', markdown.read_text()):
                url = urlsplit(link)
                if url.scheme or url.netloc or 'assets/' not in url.path:
                    continue
                source = (markdown.parent / unquote(url.path)).resolve()
                self.check(
                    source.is_relative_to(root / 'assets') and source.is_file(),
                    f'{markdown.relative_to(root)}: asset must resolve in assets/: {link}',
                )

    def _check_interactive(self) -> None:
        self.check(
            {
                'exponential/training.html',
                'cosinus/training.html',
                'laplace/training.html',
                'poiseuille/training.html',
                'cosinus_comparison/comparison.html',
                'cylinder/plots/flow-fields.html',
                'cylinder/plots/equation-residuals.html',
                'cylinder/plots/full-channel.html',
                'cylinder/plots/convergence.html',
            }.issubset(
                (
                    path.relative_to(self.site / 'assets/examples').as_posix()
                    for path in self.interactive
                )
            ),
            'Missing published training, comparison, or cylinder result figures',
        )
        for path, page in self.interactive.items():
            self.check(
                page.meta.get('robots') == ['noindex'],
                f'{path.name}: figure needs noindex',
            )
            self.check(
                'plotly.min.js' in page.links,
                f'{path.name}: missing local Plotly runtime',
            )

    def _check_content(self) -> None:
        titles = []
        descriptions = []
        verification = os.environ.get('GOOGLE_SITE_VERIFICATION', '')
        for path, page in self.content.items():
            label = path.relative_to(self.site)
            canonical = self.page_url(path)
            self.check(
                page.canonicals == [canonical], f'{label}: wrong or missing canonical'
            )
            self.check(bool(page.title.strip()), f'{label}: missing title')
            self.check(page.headings == 1, f'{label}: expected exactly one H1')
            description = page.meta.get('description', [])
            self.check(
                len(description) == 1 and bool(description[0]),
                f'{label}: missing description',
            )
            self.check(
                page.meta.get('og:url') == [canonical], f'{label}: wrong social URL'
            )
            self.check(
                bool(page.meta.get('og:title')), f'{label}: missing social title'
            )
            self.check(
                page.meta.get('og:description') == description,
                f'{label}: wrong social description',
            )
            self.check(
                not any(
                    (
                        'noindex' in value.lower()
                        for value in page.meta.get('robots', [])
                    )
                ),
                f'{label}: content must be indexable',
            )
            self.check(
                self.config.repo_url in page.links, f'{label}: missing repository link'
            )
            expected_verification = [verification] if verification else None
            self.check(
                page.meta.get('google-site-verification') == expected_verification,
                f'{label}: incorrect verification token',
            )
            if path.parent.name.endswith('pytorch'):
                self.check(page.math > 0, f'{label}: equations were not processed')
            titles.append(page.title)
            descriptions.extend(description)
        for values, label in ((titles, 'titles'), (descriptions, 'descriptions')):
            duplicates = [
                value for value, count in Counter(values).items() if count > 1
            ]
            self.check(not duplicates, f'Duplicate {label}: {duplicates}')

    def _check_tutorials(self) -> None:
        scenarios = [
            scenario for page in self.content.values() for scenario in page.scenarios
        ]
        self.check(
            Counter(scenarios)
            == Counter(
                (
                    'exponential',
                    'cosinus',
                    'laplace',
                    'poiseuille',
                    'cylinder',
                    'circular-couette',
                )
            ),
            'Each supported equation needs exactly one dedicated tutorial',
        )

    def _check_style(self) -> None:
        stylesheet = (self.site / 'assets/site/extra.css').read_text(encoding='utf-8')
        families = re.findall('font-family:\\s*"([^"]+)"', stylesheet)
        self.check(
            families == ['Manrope', 'IBM Plex Mono'],
            'The website must declare exactly the two intended font families',
        )
        for font_url in re.findall('url\\("([^"]+)"\\)', stylesheet):
            self.check(
                (self.site / 'assets/site' / font_url).is_file(),
                f'Missing font: {font_url}',
            )
        palette = dict(re.findall('--([\\w-]+):\\s*(#[\\da-fA-F]{6});', stylesheet))
        self.check(
            palette
            == {
                'ink': '#000000',
                'paper': '#ffffff',
                'taupe': '#a69f95',
                'grey': '#7a7a7a',
                'slate': '#606b73',
            },
            'The interface must use the supplied Adobe Color palette',
        )

    def _check_links(self) -> None:
        for path, page in self.pages.items():
            self.check(page.footers == 0, f'{path.name}: footers must not be rendered')
            if path not in self.interactive:
                self.check(
                    page.sidebar_modes == ['collapsed', 'hover', 'expanded'],
                    f'{path.name}: missing sidebar mode choices',
                )
                self.check(
                    page.header_menu_buttons == 0,
                    f'{path.name}: header hamburger must not be rendered',
                )
                self.check(
                    page.header_repository_links == [self.config.repo_url],
                    f'{path.name}: missing repository link in the header',
                )
            for link in page.links:
                url = urlsplit(urljoin(self.page_url(path), link))
                if (
                    url.scheme not in ('http', 'https')
                    or url.netloc != self.base.netloc
                ):
                    continue
                self.check(
                    url.path.startswith(self.base.path),
                    f'{path.name}: link escapes site: {link}',
                )
                if not url.path.startswith(self.base.path):
                    continue
                relative = unquote(url.path[len(self.base.path) :])
                target = self.site / relative
                if url.path.endswith('/'):
                    target /= 'index.html'
                self.check(
                    target.is_file(),
                    f'{path.relative_to(self.site)}: missing target: {link}',
                )
                if url.fragment and target in self.pages:
                    self.check(
                        unquote(url.fragment) in self.pages[target].ids,
                        f'{path.relative_to(self.site)}: missing anchor: {link}',
                    )

    def _check_sitemap(self) -> None:
        sitemap = ElementTree.parse(self.site / 'sitemap.xml')
        locations = [node.text for node in sitemap.findall('.//{*}loc')]
        self.check(len(locations) == len(set(locations)), 'Duplicate URLs in sitemap')
        self.check(
            set(locations) == {self.page_url(path) for path in self.content},
            'Sitemap does not match content pages',
        )
        error_page = self.pages.get(self.site / '404.html')
        self.check(
            error_page is not None and error_page.meta.get('robots') == ['noindex'],
            '404 page needs noindex',
        )

    def run(self) -> None:
        self._check_repository_assets()
        self._check_interactive()
        self._check_content()
        self._check_tutorials()
        self._check_style()
        self._check_links()
        self._check_sitemap()
        if self.errors:
            raise SystemExit('\n'.join(self.errors))
        print(
            f'Checked {len(self.content)} pages: metadata, links, assets, math, '
            'fonts, palette, verification, and sitemap.'
        )


def check_site() -> None:
    SiteChecker().run()


if __name__ == '__main__':
    check_site()
