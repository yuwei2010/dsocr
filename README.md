# dsocr

![PyPI version](https://img.shields.io/pypi/v/dsocr.svg)

OCR documents using deepseek ocr

* [GitHub](https://github.com/yuwei2010/dsocr/) | [PyPI](https://pypi.org/project/dsocr/) | [Documentation](https://yuwei2010.github.io/dsocr/)
* Created by [Wei Yu](https://audrey.feldroy.com/) | GitHub [@yuwei2010](https://github.com/yuwei2010) | PyPI [@yuwei2010](https://pypi.org/user/yuwei2010/)
* MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://yuwei2010.github.io/dsocr/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/dsocr.git
cd dsocr

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `dsocr`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

dsocr was created in 2026 by Wei Yu.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
