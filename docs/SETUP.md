# Environment setup

[Documentation index](../README.md#documentation) · [Next: workflow](WORKFLOW.md)

## Requirements

Use Git, uv and Python 3.12. uv can download Python for you. Training uses the CPU; a GPU, CUDA, a C/C++ build step and a running web server are not required. Allow disk space for catalogs, the SQLite image database, feature caches and separate model runs. Memory use grows with the number of image rows and the full 4,096-pixel matrix.

`requirements-ml.txt` pins direct dependencies. It is not a complete transitive lockfile and the project does not define a `pyproject.toml`/`uv.lock` workflow. Use `uv pip install -r requirements-ml.txt`, rather than treating it as a fully resolved file for `uv pip sync`. Activating `.venv` is optional when invoking its Python executable directly.

## Windows / PowerShell

Install uv once:

```powershell
winget install --id astral-sh.uv -e
```

Reopen PowerShell and run:

```powershell
git clone https://github.com/ignaciojjb/final_project.git
cd final_project
uv python install 3.12
uv venv --python 3.12
uv pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For an existing checkout, run `git pull origin main` from that checkout instead of cloning again. If `.venv` already works, keep it and update dependencies; do not recreate it unnecessarily. HTTPS cloning avoids requiring an SSH key. Private repositories still require a GitHub account with access.

PowerShell examples use a backtick at the end of a continued line. Do not put spaces after the backtick. You can instead paste a command as one line. Keep the terminal working directory at `final_project`; a different directory changes the interpretation of every relative data path.

The tests should finish with `OK`. The number may increase as the project evolves. Tests use temporary files that are removed after execution; seeing synthetic scores does not mean real data has been downloaded or that `artifacts/features.joblib` exists.

## Linux / macOS

After installing uv using its [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/ignaciojjb/final_project.git
cd final_project
uv python install 3.12
uv venv --python 3.12
uv pip install -r requirements-ml.txt
.venv/bin/python -m unittest discover -s tests -v
```

To adapt subsequent PowerShell commands, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`, and `\` instead of backticks for line continuation. Paths such as `datasets/images/sample.sqlite` work on both systems.

## Cloud helper

`bash setup_environment.sh` is intended for the configured cloud checkout at **`/workspace/final_project`**. It creates a venv with access to system site packages, installs the pinned dependencies and runs tests. Its absolute directory is not portable to a Windows or arbitrary Linux checkout; use the commands above locally.

No background service must be restarted. Local files and installed dependencies are distinct from live processes. Saving a cloud configuration draft, publishing a cloud snapshot and pushing code to GitHub are separate operations. This repository does not publish cloud environments automatically.

## Network access

| Destination | Purpose |
|---|---|
| `github.com` | Clone/update the repository |
| `pypi.org`, `files.pythonhosted.org` (normal PyPI sources) | Install Python packages |
| Python distribution source selected by uv | Install Python if not already present |
| `skyserver.sdss.org` | Public read-only catalog queries and optional SDSS image provider |
| `alasky.cds.unistra.fr` | Default hips2fits images, including selected high-resolution gallery cutouts |

No SciServer credentials or GitHub token are required for the public data API. An external file-download URL may require additional destination access. Proxy/firewall restrictions can differ between cloud and local machines; a successful download on one does not prove access on the other. Keep HTTPS/TLS checks enabled.

The original large CSV and ZIP attachments are not included in Git. Use the public catalog downloader or your own local files, as described in the workflow.
