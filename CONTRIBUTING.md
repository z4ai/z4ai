<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing to z4ai

Thanks for your interest in z4ai. Contributions of all kinds are welcome —
bug reports, benchmarks on new model formats, performance work, and docs.

## Reporting bugs and requesting features

Please open an issue at <https://github.com/z4ai/z4ai/issues>. For a bug, include:

- your OS, Python version, and `z4ai`, `numpy`, and `zstandard` versions;
- a minimal snippet that reproduces the problem (ideally with a small array);
- what you expected and what happened instead.

Because z4ai is a **lossless** codec, any input that does not round-trip
bit-for-bit (`z4ai.decompress(z4ai.compress(x)) == x`) is a correctness bug and
the highest priority — please report it with the failing bytes if you can.

## Seeking support

Open a [GitHub Discussion](https://github.com/z4ai/z4ai/discussions) or a
question-tagged issue. For usage, see the
[documentation](https://z4ai.github.io/z4ai/).

## Contributing code

1. Fork the repo and create a branch from `main`.
2. Set up a dev environment and run the test suite:
   ```bash
   python -m venv .venv && . .venv/bin/activate
   pip install -e ".[test,benchmark]"
   PYTHONPATH=.:tests:benchmarks python -m pytest tests/
   ```
3. Add tests for your change. Round-trip (lossless) correctness must be covered;
   ratio/throughput changes should come with a benchmark or a note in the PR.
4. Format your code with [black](https://black.readthedocs.io/) before pushing;
   CI runs `black --check` and will fail on unformatted code:
   ```bash
   pip install -e ".[lint]"
   black .          # autoformat, or `black --check .` to just verify
   ```
5. Keep the public API stable (see `API_CONTRACT.md`) and the output format
   backward-compatible, or call the break out explicitly in your PR.
6. Open a pull request describing the change and the measurements behind it.

## Code of conduct

By participating you agree to uphold a respectful, harassment-free environment.
Report concerns to the maintainer via the email in `pyproject.toml`.

## License

By contributing, you agree that your contributions are licensed under the
project's Apache-2.0 license.
