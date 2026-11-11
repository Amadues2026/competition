# Contributing

Bug reports, reproducibility improvements, tests, and focused code changes are welcome.

## Before changing code

1. Search existing issues and identify the behavior to improve.
2. For a bug, include the commit, environment, command, expected behavior, actual behavior, and a minimal reproduction.
3. Keep unrelated refactoring out of the change. Preserve public topic/message contracts unless a migration is documented.

## Local checks

Run `python -m unittest discover -s tests -v` for dataset tools and follow [Testing](docs/testing.md) for C++ checks. GPU runtime changes need a compatible runtime build and simulator/inference verification; CPU CI alone is insufficient. State which checks were run and which require another environment.

Use English for new documentation, user-visible messages, and new comments. Follow [C++ style](src/go/CODING_STYLE.md). Do not commit build/install/log directories, datasets, local credentials, editor state, or generated training output. Keep examples free of machine-specific home directories.

## Pull requests

Explain the problem and resulting behavior. Include test evidence and any limitations. Do not invent download counts, users, benchmark results, or upstream affiliations. New dependencies and imported assets need their source and license recorded in `THIRD_PARTY_NOTICES.md`.
