# Rules for Publishing

(This document is intended for maintainers)

## Rules for Publishing

Publishing is done off of the `master` branch.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), and versions should always adhere to those standards.

The versions of the Python Wheel and Rust Crate should be the same.

## Steps to Publish a New Version

1. The new version should be already committed to `master` in all `Cargo.toml` and `pyproject.toml` files
2. A tag of the form `v{VERSION}` should be pushed to the curret `master`, with `{VERSION}` matching the new version tag. Only maintainers should be able to do this.
3. The GitHub Action workflow will build and deploy to both [crates.io](https://crates.io/) and [PyPi](https://pypi.org/) automatically. The guide and other relevant documentation will also be deployed to the GitHub Pages site at [lace.dev](https://www.lace.dev/), and Python documentation will be built by [ReadTheDocs](https://readthedocs.org/)