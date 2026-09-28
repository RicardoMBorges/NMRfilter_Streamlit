# NMRfilter Streamlit v15 — Visual ranking and aligned HTML plots

Engine baseline: v14/v13 matching, parser, clustering and ranking logic preserved.

## Presentation changes
- Ranking is shown as a structured Streamlit table.
- Matching-rate percentage columns use progress bars while matched/total columns remain visible.
- Clean `result/ranking_table.tsv` is written for display/export.
- Interactive HTML candidate plots place active experiments side-by-side (HMBC | HSQC | optional HSQC-TOCSY).
- All panels share the same reversed 13C axis (0–200 ppm) for direct vertical comparison.
- Closest-unmatched markers use opacity 0.35; measured-unused markers use opacity 0.25.
- Matched markers remain fully opaque.
- HTML files remain standalone and downloadable individually or as a ZIP.

## Validation
- Parser/clustering regression suite: 9/9 passed.
- Synthetic standalone HTML test: HMBC + HSQC + matched/unmatched categories generated successfully.
