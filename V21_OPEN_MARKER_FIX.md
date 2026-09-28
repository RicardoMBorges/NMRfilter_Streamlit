# v21 — Open-marker rendering fix

Presentation-only patch.

- Matched HMBC/HSQC: solid green circles.
- Simulated HMBC/HSQC: gray open circles.
- Closest unmatched experimental: red open squares.
- Miscellaneous / measured unused: gray open squares.
- For Plotly `*-open` symbols, `marker.color` now carries the selected RGBA directly. This is required because Plotly uses marker color as the visible stroke for open symbols; a transparent marker color made the exported SVG stroke invisible even when `marker.line.color` was set.
- Sidebar opacity sliders remain the source of alpha values.
- Scientific engine, matching, clustering, ranking, and N/A semantics are unchanged.
