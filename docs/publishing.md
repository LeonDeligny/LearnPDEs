# Tutorial website: preview, launch, and discovery

Status: prepared for local preview. The site has not been deployed, the Search
Console property has not been verified, and no links or community posts have
been published as part of this change.

Planned site: <https://leondeligny.github.io/LearnPDEs/>

Planned sitemap: <https://leondeligny.github.io/LearnPDEs/sitemap.xml>

## Preview and validate without deploying

Run from the repository root:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-docs
uv run --locked --only-group docs mkdocs build --strict -f docs/website/mkdocs.yml
uv run --locked --only-group docs python docs/website/check.py
uv run --locked --only-group docs mkdocs serve -f docs/website/mkdocs.yml
```

Open <http://127.0.0.1:8000/LearnPDEs/>. This uses a separate, small environment;
it does not install the training dependencies or change the existing `.venv`.
The generated `site/` directory and `.venv-docs/` are ignored by Git.

The four tutorials and setup guide remain in `docs/tutorials/`. The build
reuses their Markdown, four GIFs, and the Plotly exports in `assets/examples/`.
A hook embeds the interactive figures and adapts repository links for the website,
so GitHub's Markdown links and GIFs still work. The Plotly runtime is served locally;
the figures require JavaScript but no external service or training server.
Add a unique `title` and `description` to each page's YAML front matter and
include new pages in `docs/website/mkdocs.yml`. The build rejects missing metadata
or broken Markdown links; `docs/website/check.py` checks the generated artifact too.

The site includes canonical URLs, social sharing metadata, a sitemap, local
search, and math rendering. MathJax is loaded from a versioned CDN URL; equations
need network access to that CDN. The written tutorials and code are static HTML.
The sitemap omits optional modification dates so rebuilds do not imply that
every tutorial's content changed.

The interface uses exactly two locally hosted font families: Manrope for the
interface and reading text, and IBM Plex Mono for code. Font licenses are in
`assets/site/fonts/`. Icons are inline SVG. MathJax uses SVG output so equations
do not request another font family. The site has a custom theme with no
Bootstrap or icon-font dependencies.

The minimal interface uses the supplied Adobe Color palette: black `#000000`,
white `#FFFFFF`, taupe `#A69F95`, grey `#7A7A7A`, and slate `#606B73`.
Light surfaces mix these swatches with white; slate keeps small text readable.
Training figures stay collapsed until opened.

The bottom-left sidebar control offers collapsed, expand-on-hover, and expanded
modes. The choice is stored locally across pages. Without a saved preference,
small screens use an icon rail; larger screens use expanded navigation. Hover
mode also expands on keyboard focus or a first tap on wider touchscreens.
Expansion moves content to the right rather than covering it. At 760px or
narrower, all modes automatically use the rail; the saved preference returns
when space permits. No header hamburger is used.

Each tutorial covers one problem: exponential growth, the harmonic
oscillator, Laplace, or Poiseuille flow. Initial-value, slope, and boundary-loss explanations
are included in the relevant equation's tutorial. New tutorial pages need the
same front-matter fields as these pages; the sidebar and dashboard cards use
that metadata directly.

## Deployment controls

`.github/workflows/docs.yml` builds and checks changes on pushes and pull
requests. Neither event can deploy. A manual run also defaults to build only.
Uploading a Pages artifact and deploying both require all three conditions:

1. The event is `workflow_dispatch`.
2. The `deploy` input is explicitly selected.
3. The selected branch is `main`.

No workflow dispatch, Pages settings change, or deployment is needed for local
preview. The following launch steps are deferred until publication is wanted.

## Launch later

1. Keep `site_url` in `docs/website/mkdocs.yml` aligned with the eventual public
   address. For the current repository, the project URL is
   `https://leondeligny.github.io/LearnPDEs/`. If using a custom domain, update
   this setting, the links below, and the Search Console property together.
2. In [Google Search Console](https://search.google.com/search-console/), add a
   **URL-prefix** property for that complete URL, including `/LearnPDEs/`.
   Select **HTML tag** verification and copy only the `content` value. You do
   not control DNS for `github.io`, so use the URL-prefix method for this host.
3. In the repository's **Settings → Secrets and variables → Actions → Variables**,
   add `GOOGLE_SITE_VERIFICATION` with that value. It is a public verification
   token, not a Google password or an OAuth credential. Keep the variable after
   verification: Google may check the tag again.
4. Save these changes to `main` when ready. In **Settings → Pages → Build and
   deployment**, select **GitHub Actions**. Configure the `github-pages`
   environment's allowed deployment branch as `main`; required reviewers can
   also be added if available for the repository.
5. In **Actions → Documentation → Run workflow**, choose `main`, select
   **Publish the built documentation to GitHub Pages**, and run it. Wait for
   the build, artifact checks, and deployment to succeed.
6. Open the live home page, every tutorial, and `sitemap.xml`. Check navigation,
   equations, images, page source, and the `google-site-verification` meta tag
   in the home page's `<head>`. A missing URL should return HTTP 404.
7. Back in Search Console, click **Verify**. Adding the tag locally or setting
   the variable alone does not verify ownership; Google must fetch it live.

To test a real token locally before launch:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-docs
export GOOGLE_SITE_VERIFICATION='paste-the-content-value-from-google'
uv run --locked --only-group docs mkdocs build --strict -f docs/website/mkdocs.yml
uv run --locked --only-group docs python docs/website/check.py
```

Sources: [GitHub Pages overview](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages),
[Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages),
and [Google's ownership verification instructions](https://support.google.com/webmasters/answer/9008080).

## Submit the sitemap and check indexing after verification

In the verified URL-prefix property, open **Sitemaps**, submit
`https://leondeligny.github.io/LearnPDEs/sitemap.xml`, and record the submission
date and status in the tracking table below. The sitemap should contain six
URLs: home, setup, and the four tutorials. It excludes the error page.

Use **URL Inspection** on the home page and tutorial URLs. Run a live test if
needed, check Google's access and selected canonical, then request indexing
for these few pages. This requires an owner or full user of the verified
property. A request does not guarantee inclusion; crawling can take days or
weeks, and repeated requests do not make the same URL crawl faster.

Do not use Google's Indexing API for these tutorials; Google's
[Indexing API scope](https://developers.google.com/search/apis/indexing-api/v3/using-api)
is limited to supported job posting and livestream pages.

There is deliberately no project-level `robots.txt`: Google only reads it at
the host root, `https://leondeligny.github.io/robots.txt`, not underneath
`/LearnPDEs/`. Check that host-root file at launch if one exists. If you manage
that separate user-site repository, it can advertise the full sitemap URL and
must not disallow this project. Direct sitemap submission works without it.

Sources: [requesting a recrawl](https://developers.google.com/search/docs/crawling-indexing/ask-google-to-recrawl),
[building and submitting a sitemap](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap),
and [robots.txt location and scope](https://developers.google.com/search/docs/crawling-indexing/robots/intro).

## Monitor indexing and useful search queries

Check weekly for the first month after launch, then monthly. Search Console
does not require adding visitor analytics to the site.

| Check | What to record | Follow-up |
| --- | --- | --- |
| Sitemaps | Last read, fetch status, discovered URLs | Fix fetch or XML errors; confirm the production URL. |
| Page indexing | Indexed pages and reasons for exclusions | Inspect affected URLs; check HTTP responses, canonical URLs, and crawl access. |
| Performance → Search results | Date range, clicks, impressions, CTR, average position | Compare complete 28-day periods, filtering by tutorial page. |
| Queries and pages | Which questions lead to which tutorials | Improve explanations and link text where readers' needs fit the content. |

Early reports may be empty. Zero impressions do not by themselves prove an
indexing problem; inspect the URL. Some queries are omitted for privacy, so
query rows may not add up to overall totals. Use the report's export option to
save comparisons; never replace empty data with assumed rankings.

| Review date | Sitemap status / last read | Indexed / expected | Period | Clicks | Impressions | CTR | Avg. position | Query/page findings and next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pending launch | Not submitted | Unknown / 5 | — | — | — | — | — | Verify the live property first. |

Sources: [Page indexing report](https://support.google.com/webmasters/answer/7440203)
and [Search performance report](https://support.google.com/webmasters/answer/7576553).

## Links and community sharing after launch

Only share URLs once they resolve publicly. Use the most relevant tutorial,
describe what it teaches, and disclose that LearnPDEs is your project.
Google discovers pages through links, but links and sharing do not guarantee
rankings. See [Google's discovery and promotion guidance](https://developers.google.com/search/docs/fundamentals/seo-starter-guide).

For a personal site's projects or teaching page, this snippet is ready to adapt:

```markdown
[LearnPDEs: physics-informed neural networks in PyTorch](https://leondeligny.github.io/LearnPDEs/)
— my worked tutorials on exponential growth, the harmonic oscillator, and the
Laplace equation, with runnable examples and analytical error checks.
```

For a project writeup about automatic differentiation or scientific computing:

```markdown
I worked through [solving the Laplace equation with a PINN in PyTorch](https://leondeligny.github.io/LearnPDEs/tutorials/laplace-pinn-pytorch/),
including the two-dimensional residual, four Dirichlet boundaries, and
comparison with an analytical solution. The [code is available in LearnPDEs](https://github.com/LeonDeligny/LearnPDEs).
```

Community post draft:

> I maintain LearnPDEs, a small PyTorch project for learning physics-informed
> neural networks. I've written separate worked tutorials for exponential
> growth, the harmonic oscillator, and the Laplace equation on a square. They
> include CPU-runnable examples and compare predictions with analytical
> solutions. One lesson from the ODE example is that a small residual at the
> training points can still miss a large error between them. I would welcome
> feedback on the derivations and evaluation checks.
>
> https://leondeligny.github.io/LearnPDEs/tutorials/

Suggested audiences are scientific-computing reading groups, educational
PyTorch discussions, and PINN or scientific-machine-learning communities you
already participate in. Check the destination's current rules and choose a
resources/showcase area that accepts project sharing. Share a focused answer
with a supporting tutorial link when a real discussion calls for it.

| Placement | Target page | Status | Published URL / date |
| --- | --- | --- | --- |
| Personal website's projects/teaching page | Home | Draft; personal site location needed | — |
| Related scientific computing writeup | Laplace tutorial | Draft; writeup location needed | — |
| Relevant community resources discussion | Setup or a specific tutorial | Draft; choose an appropriate venue after launch | — |

The site configuration excludes this publishing guide, `docs/README.md`, and
`docs/website/`, so maintainer notes and build files are not published or indexed.
