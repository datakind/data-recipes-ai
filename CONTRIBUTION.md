# Development

## Pre-commit hooks

The repo has been set up with black and flake8 pre-commit hooks. These are configured in the ``.pre-commit-config.yaml` file and initialized with `pre-commit autoupdate`.

On a new repo, you must run `pre-commit install` to add pre-commit hooks.

To run code quality tests, you can run `pre-commit run --all-files`