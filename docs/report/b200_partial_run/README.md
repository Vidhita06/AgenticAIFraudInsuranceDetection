# Partial B200 run (superseded)

These files were pushed to the root of `main` (commit d80babe) and moved here when PR #2 was
merged. They come from an **earlier, incomplete B200 run**, not from the run that produced
the artifacts in `models/`:

- `lightgbm` and `catboost` were not installed on the server, and the MLP failed with
  "Numpy is not available", so only LogReg, RF and XGBoost were trained.
- The data-mix experiment chose w = 0.3. The final run chose w = 0.1.
- The log stops at the start of XGBoost tuning.

They are kept for reference. The final model and its results are described in
`docs/report/02_model_report.md` and `docs/report/model_card.md`.
