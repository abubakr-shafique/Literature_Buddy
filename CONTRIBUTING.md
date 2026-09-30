# Contributing

1. Fork, create a branch, `pip install -r requirements-dev.txt && pip install -e . --no-deps`.
2. Keep `ruff check` clean and `QT_QPA_PLATFORM=offscreen pytest` green; add tests for behaviour changes.
3. Parsing bugs: attach the PDF (or a minimal reproduction) and the relevant part of `document.json`.
4. By contributing you agree your work is licensed under AGPL-3.0-or-later.
