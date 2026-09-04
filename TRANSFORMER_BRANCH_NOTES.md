# transformer branch -- isolated environment notes

Genuine clone of the Ubuntu Dialogue Corpus repo (full history, `origin` =
https://github.com/lilywool/Ubuntu-dialogue-corpus.git), checked out on its
own `transformer` branch, per project isolation rules: separate clone,
separate `.venv`, own pinned Python/PyTorch/Transformers/CUDA versions,
never shared with the Ubuntu `master` clone or the Yelp repo.

## Status as of 2026-09-04

- [x] Genuine clone with full history (not a scaffold with one copied file)
- [x] `transformer` branch created
- [x] `requirements.txt` pinned: PyPI as primary index, PyTorch CPU repo as
      an *extra* index only, `torch==2.5.1+cpu` pinned to the explicit local
      build tag so it can't silently resolve a CUDA wheel
- [x] Environment mechanism decided: a **prefix-based Conda environment**
      created at `.venv` via the existing Anaconda installation's `conda`
      executable -- no global Python install, no PATH/shell-profile
      changes, no changes to Anaconda's base env, env stays inside this
      repo. Conda's shared package cache is fine (isolation rules allow
      shared *caches*, just not shared *environments*).
- [ ] **Not yet run** -- I can't execute native Windows commands from this
      bridge (it only reaches a Linux VM on this machine). Commands below
      need to be run by hand in PowerShell.
- [ ] GPU presence is unconfirmed (not "no GPU") -- see requirements.txt
      comment.
- [ ] No dtype/CUDA-related code changes made yet to `sentiment_analysis.py`
      in this branch -- unmodified copy, carried over via the clone.

## Setup (run from this folder in PowerShell)

First confirm where conda.exe actually lives (path below is the typical
default install layout -- adjust if different):

    & "C:\Users\lilli_37fwt34\anaconda3\Scripts\conda.exe" --version

Then create the prefix env and install pinned deps -- this only touches
`.\.venv` inside this repo, nothing global:

    & "C:\Users\lilli_37fwt34\anaconda3\Scripts\conda.exe" create --prefix ".\.venv" python=3.11 -y
    .\.venv\python.exe -m pip install --upgrade pip
    .\.venv\python.exe -m pip install -r requirements.txt
    .\.venv\python.exe -m pip freeze > requirements.lock.txt

Note the Windows conda-env layout: `python.exe` sits directly at the
prefix root (`.\.venv\python.exe`), not under `Scripts\` -- `Scripts\`
holds `pip.exe` and other console-script shims instead.

Then reproduce the transformer behavior on a small deterministic sample
(step 2 of the agreed plan) before touching any dtype/validation code.
