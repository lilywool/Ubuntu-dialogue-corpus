# transformer branch -- isolated environment notes

This is a genuine clone of the Ubuntu Dialogue Corpus repo (full history,
`origin` = https://github.com/lilywool/Ubuntu-dialogue-corpus.git), checked
out on its own `transformer` branch, per project isolation rules: separate
clone, separate `.venv`, own pinned Python/PyTorch/Transformers/CUDA
versions, never shared with the Ubuntu `master` clone or the Yelp repo.

## Status as of 2026-09-04

- [x] Genuine clone with full history (not a fresh scaffold with one file)
- [x] `transformer` branch created
- [x] `requirements.txt` pinned, with explicit `--index-url` for the CPU
      torch wheel (not relying on a bare version number)
- [ ] **Blocked:** `.venv` not yet created. The Windows `py` launcher is not
      installed, and no standalone Python 3.11 interpreter has been located
      -- only Anaconda's Python 3.13.5. Installing a Python 3.11 interpreter
      requires explicit approval before I do it or hand over install steps.
- [ ] GPU presence is unconfirmed (not "no GPU") -- see requirements.txt
      comment. Worth an explicit native-Windows check (e.g. Device
      Manager / `Get-CimInstance Win32_VideoController` in PowerShell)
      before assuming CPU-only is permanent.
- [ ] No dtype/CUDA-related code changes have been made yet to
      `sentiment_analysis.py` in this branch -- it's an unmodified copy,
      carried over from `master` via the clone.

## Next steps (once Python 3.11 is approved/available)

    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m pip freeze > requirements.lock.txt

Then reproduce the transformer behavior on a small deterministic sample
(step 2 of the agreed plan) before touching any dtype/validation code.
