# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).

## Working on this fork

The upstream notices and contribution policies above are retained. This fork's
runtime is NumPy/Gymnasium with optional PyTorch agents. Use Python 3.12–3.14 and
install the workspace rather than the historical `recsim` PyPI package.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip check
python -m pytest -m 'not rl'
python scripts/check_no_legacy_imports.py
```

Verify the base environment without Torch before installing learning dependencies:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[rl,dev,docs]'
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy recsim
python scripts/run_notebooks.py
python scripts/generate_api_docs.py
python -m build
```

Notebook execution starts local Jupyter kernels and requires local socket access.
Source notebooks keep cleared outputs. API Markdown is generated from public
symbols and docstrings; edit the Python source or generator, then regenerate.
The generator also retains explanatory pages for removed historical symbols.

For an installation check, install the built wheel in a separate virtualenv,
change to a directory outside this checkout, and run the absolute path to
`scripts/verify_install.py --base-only`. After installing CPU Torch, repeat with
`--rl`. The CI package job also builds a wheel from the source distribution.

Keep model-equation changes distinct from API/RNG fixes in the description.
Add numerical reference or behavioral regression tests for substantive changes;
avoid testing implementation details alone. Record intentional compatibility
changes in [migration.md](docs/migration.md), and actual validation results in
[validation.md](docs/refactoring/validation.md). Never commit virtualenvs, bytecode,
training checkpoints, generated data, or notebook outputs. Dependency snapshots
are in [constraints](constraints/README.md).
