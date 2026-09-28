# v13 — legacy two-column spectrum mode

NMRfilter v1.5's parser assigns experiment identity from one-column section markers (HMBC/HSQC/HSQCTOCSY), while its README describes a two-column TAB peak list. v13 resolves this ambiguity explicitly in the Streamlit UI.

For unlabeled numeric two-column rows choose one interpretation:
- Reject as ambiguous (default; safest)
- HMBC
- HSQC
- HSQC-TOCSY

Explicit section markers or a third type column always take precedence. Mixed-experiment files must remain explicitly labeled; v13 never infers HMBC vs HSQC from chemical shifts alone.
