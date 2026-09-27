# Shared assets and published results

This directory is the single source for assets displayed in the repository and
documentation website. Markdown links point here directly; the website build
copies these files into the generated, ignored `site/` directory. Never maintain
another copy under `docs/` or edit the generated site.

- `examples/`: selected figures and their provenance, shared by README and docs.
- `verification/`: independent evaluation reports, identified by scenario and run.
- `site/`: website CSS, JavaScript, icons, and fonts.
- `runs/` and `comparisons/`: local working outputs, ignored by Git.

All training entry points use this directory as their default output root.
Passing `--output-dir /some/folder` changes the root, while keeping the same layout.

```text
assets/
  runs/
    kovasznay/
      latest.json
      <UTC timestamp>-<unique suffix>/
        run.json
        loss.csv
        collocation.json
        model.pt
        training.gif
        training.html
        plotly.min.js
        frames/
  comparisons/
    cosinus_derivatives/
      <run-id>/
        run.json
        comparison.html
        results.json
        summary.csv
        runs/cosinus/<individual-run-id>/...
  examples/
    laplace/
      training.gif
      training.html
      plotly.min.js
      run.json
    cylinder/
      plots/
        diagnostics.json
        flow-fields.html
        flow-fields.png
        equation-residuals.html
        full-channel.html
        convergence.html
        plotly.min.js
  verification/
    cylinder/<run-id>.json
  site/
    extra.css
    fonts/
```

## Working runs

Every invocation reserves a new directory. UTC timestamps help sorting; a random
suffix and exclusive directory creation prevent collisions. Repeating a scenario,
seed, or output root never replaces earlier results. The console prints the actual
GIF, HTML, and manifest paths. `latest.json` points to the last successfully
completed run for that scenario; it is a convenience pointer, not a history file.

`run.json` has a versioned schema and records effective settings, seed, network
architecture, sample counts/bounds, optimizer, export settings, library versions,
source commit and whether the working tree was modified. CLI runners also record
their independent validation errors. A dirty working tree is recorded honestly;
a commit identifier alone cannot reproduce uncommitted code.

The run lifecycle is `created → training → exporting → completed`. Exceptions
produce `failed`, and Ctrl-C produces `interrupted`, with an error and the number
of completed updates. A forcibly killed process can leave a `training` or
`exporting` record; those states never imply success. JSON updates are atomic.

`loss.csv` records each evaluated step, including step zero and the final update.
`model.pt` contains model and Adam state dictionaries and the completed step count.
These are saved **before rendering**. Load on another device with
`torch.load(path, map_location='cpu', weights_only=True)` and reconstruct the
network using the recorded settings. This is a checkpoint, not a promise of an
exactly reproducible restart across different hardware or libraries.

`collocation.json` records the actual loss coordinates at each animation
checkpoint, independently of the display and validation grids. Its versioned
schema stores `point_sets` once and maps each `checkpoints` entry's `step` to a
`point_set` index. Groups identify PDE, boundary, initial/pressure-anchor, and flux
quadrature roles; overlapping coordinates retain each applicable role. Recording
is limited to displayed checkpoints, not every optimizer update. GIFs show these
points; HTML also offers visibility controls. Older runs without this file are
reported as missing coordinates, rather than reconstructed from new samples.

GIFs include initial and final states, loop at 10 FPS, and hold the last frame for
two seconds. `--no-gif` omits GIF/PNG rendering. PNGs live only in that run's
`frames/` directory and are retained, including when rendering or encoding fails.
Successful runs include SHA-256 hashes and sizes of their final artifacts.

Each HTML figure carries its own matching `plotly.min.js`. Copy the whole run
directory to share it offline. Runs and comparisons are ignored by Git to keep
large experiments out of routine commits. Delete an entire unwanted run directory
when reclaiming space; there is no automatic retention policy.

## Published examples

`examples/` contains deliberately selected, version-controlled results used by the
README and website. Training never updates these implicitly. To publish a completed
run after reviewing its accuracy:

```bash
uv run learnpdes publish 'assets/runs/laplace/<run-id>'
# Explicitly update an existing example:
uv run learnpdes publish 'assets/runs/laplace/<run-id>' --replace
```

Publication checks hashes, stages copies, and uses a lock to reject simultaneous
publishers. A failed copy leaves the previous example intact. It includes the
figures, their runtime, loss history, recorded coordinates, and run metadata; weights and PNG frames stay
with the working run. Only the selected example needs to be committed.
Replacing an example that includes a GIF requires a new GIF too, so publication
cannot silently break the README's image links or mix figures from different runs.

For cylinder diagnostic plots, generate them inside the completed run and publish
them together. Use the same asset root for verification and publication:

```bash
uv run learnpdes verify 'assets/runs/cylinder/<run-id>'
uv run learnpdes plot 'assets/runs/cylinder/<run-id>' --png
uv run learnpdes publish 'assets/runs/cylinder/<run-id>' --replace
```

Publication includes `plots/` HTML, standalone PNGs, their runtime, and
`diagnostics.json`. It checks the plots' run ID and checkpoint hash and records
each file's checksum in the published manifest. Replacing an example must retain
its existing plot filenames so documentation links stay valid. The website reads
the cylinder metrics from the matching report in `verification/` and rejects
plots and reports from different checkpoints.

Existing tutorial figures were moved without retraining. Their records use
`status: imported`, retain separate GIF/HTML provenance where the old exporters
used different runs, and do not invent missing weights or settings. Original
manifests are retained under `examples/_provenance/`. Historical local outputs are
preserved in `runs/<scenario>/imported-*` and
`comparisons/cosinus_derivatives/imported/`.

The documentation build reads `examples/`, `verification/`, and `site/`.
It excludes archived `_provenance/` files and never publishes working runs.
