# v16 — Not-evaluated experiment semantics

This release preserves the v15 numerical engine and changes reporting semantics only.

- An experiment is evaluable only when measured peaks of that experiment are present in `realspectrum.csv`.
- If no measured peaks are present, its matching rate is reported as `N/A`, not `0%`.
- `0%` is reserved for an experiment that was provided and evaluated but yielded zero matches.
- Interactive HTML panels remain visible for predicted correlations, but absent experimental evidence is marked `Not evaluated`.
- This behavior is symmetric for HMBC, HSQC, and HSQC-TOCSY.
