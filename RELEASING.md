# Releasing

How to cut a new release of `terminalboard` to [PyPI](https://pypi.org/project/terminalboard/).

The version is **single-sourced** from `terminalboard/__init__.py` (`__version__`);
`pyproject.toml` reads it dynamically, so you bump it in **one place**.

PyPI versions are **immutable** — you can never re-upload or overwrite a version
that already exists. Every release needs a new, higher version number
(roughly [SemVer](https://semver.org/)): `0.1.1` for a fix, `0.2.0` for features,
`1.0.0` for a stable API.

## 1. Bump the version

Edit the one line in `terminalboard/__init__.py`:

```python
__version__ = "0.1.1"
```

Commit it:

```bash
git commit -am "Release v0.1.1"
git push
```

## 2a. Publish manually (API token)

```bash
pip install --upgrade build twine     # once
rm -rf dist                           # IMPORTANT: avoid re-uploading old files
python -m build                       # builds sdist + wheel into dist/
twine check dist/*                    # validate metadata / README rendering
twine upload dist/*                   # uses ~/.pypirc (username __token__)
```

`~/.pypirc` should hold a **project-scoped** token:

```ini
[pypi]
  username = __token__
  password = pypi-...          # token scoped to the "terminalboard" project
```

Optional dry run on [TestPyPI](https://test.pypi.org/) first:

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ terminalboard
```

## 2b. Publish automatically (GitHub Actions, no token)

The repo ships `.github/workflows/publish.yml`, which builds and publishes via
PyPI **Trusted Publishing** (OIDC) whenever a `v*` tag is pushed.

One-time PyPI setup — **Your account → Publishing → Add a publisher**:

| Field | Value |
|---|---|
| PyPI Project Name | `terminalboard` |
| Owner | `yexLing` |
| Repository name | `terminalboard` |
| Workflow name | `publish.yml` |
| Environment | `pypi` |

Then, after step 1 (version bump pushed):

```bash
git tag v0.1.1
git push --tags
```

The tag push triggers the workflow, which builds and uploads automatically — no
secrets stored. Optionally create a GitHub Release from the tag afterwards for
release notes.

## 3. Verify

```bash
python3 -m venv /tmp/t && /tmp/t/bin/pip install terminalboard
/tmp/t/bin/terminalboard --version       # should print the new version
```

Then check the project page: <https://pypi.org/project/terminalboard/>.
