# Documentation

Start with the [tutorial setup guide](tutorials/README.md), then choose a
[worked tutorial](tutorials/README.md#choose-an-equation).

The [validation plan](validation.md) defines the no-simulation-data rule for
every scenario, the gradual ODE/PDE progression, papers and exact comparison
values, and the acceptance gates required before increasing complexity.

All documentation content and website tooling live here:

```text
docs/
├── index.md           Website home page
├── tutorials/         Setup, exponential growth, oscillator, Laplace, Poiseuille
├── publishing.md      Preview, launch, Search Console, and sharing guide
└── website/           MkDocs configuration, checks, hooks, and templates
```

The repository's [`assets/`](../assets/README.md) directory is the single source
for displayed figures, verification reports, and website resources:

- `assets/examples/` holds selected training figures and cylinder diagnostic plots.
- `assets/verification/` holds independent reports for the displayed checkpoints.
- `assets/site/` holds styles, scripts, icons, and the two locally hosted fonts.

The project README and tutorials link to these files directly. Training examples
contain their Plotly HTML figure, local JavaScript runtime, and run metadata. The site embeds
these interactive figures and offers GIF links; GitHub Markdown keeps the GIFs.
Generate HTML-only runs with `uv run learnpdes train all --no-gif`,
or HTML and GIFs with `uv run learnpdes train all`.
To replace a recorded example, publish a selected run with
`uv run learnpdes publish <run-directory> --replace`. The site
build copies the selected assets into the ignored `site/` output, so its small
docs environment needs neither PyTorch nor Plotly. Interactive figures work offline
and start paused.
Keep assets in the repository-level directory; `docs/assets/` is rejected by the
build to prevent a second source of results.
GitHub's workflow stays in `.github/workflows/docs.yml`, where Actions requires it.

To preview from the repository root:

```bash
UV_PROJECT_ENVIRONMENT=.venv-docs uv run --locked --only-group docs mkdocs serve -f docs/website/mkdocs.yml
```

Open <http://127.0.0.1:8000/LearnPDEs/>. See [publishing.md](publishing.md)
for build checks and the deferred launch steps. Nothing deploys automatically.
