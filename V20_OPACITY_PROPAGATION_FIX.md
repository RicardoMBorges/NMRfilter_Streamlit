# v20 — Plot opacity propagation fix

- Sidebar opacity sliders are passed through project properties to the HTML plot generator.
- Opacity is now encoded directly into the RGBA alpha of marker fill/outline.
- Open markers therefore obey the selected opacity on their visible border.
- Each HTML title records the four opacity values actually used.
- Sidebar shows the four values that will be used on the next run.
- Numerical matching, clustering and ranking are unchanged.
