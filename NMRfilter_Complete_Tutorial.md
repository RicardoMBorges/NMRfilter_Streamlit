# NMRfilter on Streamlit: Complete User Tutorial

**Compare candidate structures with experimental 2D NMR data, inspect the ranking, and explore complete atomic chemical shift predictions.**

- **Online application:** [nmrfilter.streamlit.app](https://nmrfilter.streamlit.app/)
- **Source repository:** [RicardoMBorges/NMRfilter_Streamlit](https://github.com/RicardoMBorges/NMRfilter_Streamlit)
- **Example files:** [Mock data](https://github.com/RicardoMBorges/NMRfilter_Streamlit/tree/main/mock_data)
- **Video:** [NMRfilter video tutorial](https://www.youtube.com/watch?v=pkY-rmvfDdU)

Tutorial edition: October 2026. This guide covers the Streamlit workflow and the extensions for atomic exports, HOSE diagnostics, and structure visualization. These extensions require the corresponding updated application version; an older deployment may show fewer controls.

## Contents

1. [What NMRfilter does](#1-what-nmrfilter-does)
2. [Quick start](#2-quick-start)
3. [Prepare the experimental data](#3-prepare-the-experimental-data)
4. [Prepare candidate structures and names](#4-prepare-candidate-structures-and-names)
5. [Prepare the measured spectrum file](#5-prepare-the-measured-spectrum-file)
6. [Configure the analysis](#6-configure-the-analysis)
7. [Run the pipeline](#7-run-the-pipeline)
8. [Understand experimental clustering](#8-understand-experimental-clustering)
9. [Interpret the ranking](#9-interpret-the-ranking)
10. [Read the interactive spectra](#10-read-the-interactive-spectra)
11. [Explore structures and atomic shifts](#11-explore-structures-and-atomic-shifts)
12. [Download and navigate the outputs](#12-download-and-navigate-the-outputs)
13. [Understand the atomic prediction tables](#13-understand-the-atomic-prediction-tables)
14. [Use HOSE diagnostics](#14-use-hose-diagnostics)
15. [Contribute experimental assignments to nmrshiftdb2](#15-contribute-experimental-assignments-to-nmrshiftdb2)
16. [Build confidence in a candidate](#16-build-confidence-in-a-candidate)
17. [Troubleshooting](#17-troubleshooting)
18. [Reproducibility and reporting](#18-reproducibility-and-reporting)
19. [Local execution and deployment notes](#19-local-execution-and-deployment-notes)
20. [Frequently asked questions](#20-frequently-asked-questions)
21. [References and contact](#21-references-and-contact)

## 1. What NMRfilter does

NMRfilter compares predicted NMR correlations from a supplied list of molecular structures with experimental 2D NMR peak coordinates. The Streamlit interface provides access to the original NMRfilter v1.5 workflow, together with interactive plots and additional prediction inspection tools.

The workflow combines:

1. Conversion of candidate SMILES into molecular structures.
2. Chemical shift prediction using a HOSE code reference database.
3. Simulation of candidate 2D correlations.
4. Grouping of experimental correlations into a graph and detection of communities.
5. Assignment of simulated correlations to experimental peaks and candidate ranking.

The updated interface also exports complete predicted **¹³C and ¹H atomic shifts**, including atoms that do not appear in the simulated HSQC or HMBC correlation lists. It can display those predictions on molecular structures and flag atoms with limited HOSE database support.

### The scientific question

The application asks: **Which of the supplied candidate structures are most compatible with the experimental NMR evidence?**

It is useful for candidate screening and dereplication in mixtures. Candidates may come from MS-based annotation, databases, literature, or a curated list. Their origin does not change the need to verify their structures and experimental compatibility.

A high rank is a reason to examine a candidate closely. It is not a probability of identity, a validated confidence level, or proof that the compound is present. NMRfilter cannot recover a correct structure that is absent from the candidate list, and compatible substructures can produce matches for different molecules.

### What you upload

You upload a list of structures and a list of experimental **peak coordinates in ppm**. This workflow does not require raw Bruker data and does not perform raw NMR processing, phasing, baseline correction, or peak picking for you.

## 2. Quick start

For your first run, use the files in the repository's **Mock data** folder as a consistent example set. Keep their candidate and name files together.

1. Open [NMRfilter](https://nmrfilter.streamlit.app/).
2. Expand the sidebar if it is collapsed.
3. Enter a descriptive **Project name**.
4. Upload **Candidate structures — SMILES, one per line**.
5. Upload **Measured 2D NMR spectrum — 13C and 1H shifts**.
6. Upload **Candidate names**, if supplied with the example.
7. Select the experimental solvent.
8. Check that every experimental peak has the correct experiment identity. For a file containing only two numeric columns, select its experiment under **Two-column spectrum interpretation**.
9. Start with the default clustering parameters: **¹³C tolerance = 0.2 ppm**, **¹H tolerance = 0.02 ppm**, and **Louvain/RBER resolution = 0.2**.
10. Leave **Generate interactive HTML plots** enabled. The default is the top 10 candidates.
11. Click **Run NMRfilter** in **Analysis and results**, or in the main analysis area of an older version.
12. Review the ranking, then inspect several leading candidates in the interactive plots.
13. Open **Structures and atomic shifts** to inspect atomic predictions.
14. Download the ranking, calculated NMR data, and complete project/results ZIP.

Save the results while the session is available. A browser session is not a permanent analysis archive.

## 3. Prepare the experimental data

### Recommended evidence

HSQC and HMBC provide complementary information:

| Experiment | Main information | Role in candidate assessment |
| --- | --- | --- |
| HSQC | Directly bonded proton–carbon correlations | Tests compatibility with protonated carbon environments |
| HMBC | Longer-range proton–carbon correlations | Helps distinguish connectivity patterns and candidate substructures |
| HSQC-TOCSY | Correlations extending through proton spin systems | Adds evidence when this experiment was acquired and is enabled |

HSQC generally gives a more direct connection between a proton and its attached carbon. HMBC adds connectivity evidence, but the visibility of a long-range correlation depends on coupling, pulse sequence settings, sensitivity, and the molecular environment.

### Before exporting a peak list

1. Process and reference each spectrum in your NMR software.
2. Check that the HSQC and HMBC spectra use compatible chemical shift references.
3. Select credible cross-peaks. Review solvent signals, noise, artifacts, and duplicated picks.
4. Export the **carbon shift first** and the **proton shift second**.
5. Retain experiment identity for every row.
6. Use ppm values rather than point indices, frequencies in Hz, or intensities.

A clean peak list usually helps more than an indiscriminately large one. Dense noise and duplicate peaks can alter graph connectivity and create accidental matches.

The application uses coordinate pairs. Peak intensity is not an input to the matching rate or ranking described here. Avoid including peak IDs or intensities before the two required coordinate columns.

## 4. Prepare candidate structures and names

### Candidate structure file

Use a UTF-8 plain-text `.smi` file with **one SMILES per line**. The uploader also accepts `.txt` and `.csv`, but accepting a file extension does not make an arbitrary multi-column CSV a valid structure list.

Example `candidates.smi`:

```text
CCO
CC(=O)O
COc1ccccc1
```

These are format examples, not a demonstration dataset for a particular spectrum.

For reliable input:

- Do not add a column header such as `SMILES`.
- Do not include spreadsheet row numbers or surrounding quotation marks.
- Avoid blank lines within the list. Some legacy stages stop reading at a blank line even though the atomic exporter ignores empty lines.
- Check valence, charge, aromaticity, stereochemistry, salts, and disconnected fragments.
- Remove duplicate candidates unless there is a specific reason to retain them.
- Keep a stable copy of the exact submitted list.

The updated atomic exporter can read an inline name after a SMILES. A separate names file is preferable for consistent naming across the legacy ranking and newer exports.

### Candidate names file

Provide one name per candidate, in the **same order** as the SMILES file.

Example `candidate_names.txt`:

```text
Ethanol
Acetic acid
Anisole
```

| Candidate position | SMILES | Name |
| --- | --- | --- |
| 1 | `CCO` | Ethanol |
| 2 | `CC(=O)O` | Acetic acid |
| 3 | `COc1ccccc1` | Anisole |

Do not sort the names independently. A chemically valid structure with an incorrect name can produce plausible-looking results assigned to the wrong compound.

In the atomic export, naming follows this priority: the optional names file, an inline SMILES name, then `candidate_N`. Candidate IDs refer to input order. **Input candidate ID and ranking position are different identifiers.**

## 5. Prepare the measured spectrum file

The parser accepts tab, comma, or semicolon delimiters, as well as legacy section labels. Use a decimal point for the clearest and most portable format.

### Option A: Three columns with explicit experiment labels

This is the clearest format for mixed experiments:

```csv
13C,1H,type
55.20,3.75,HSQC
112.40,6.82,HSQC
148.10,6.82,HMBC
130.50,3.75,HMBC
```

The first two columns must contain carbon and proton shifts, in that order. The third contains the experiment type.

Recognized experiment identities are **HMBC**, **HSQC**, and **HSQCTOCSY**. The input parser also normalizes forms such as `HSQC-TOCSY`.

The example coordinates above illustrate syntax only. They are not experimental evidence for the example candidate structures in Section 4.

### Option B: Legacy sectioned file

```text
HMBC
148.10 6.82
130.50 3.75
HSQC
55.20 3.75
112.40 6.82
```

Each experiment name starts a section. Rows continue to belong to that experiment until the next section label. Tabs can replace the spaces in this example.

### Option C: A two-column list for one experiment

```csv
13C,1H
55.20,3.75
112.40,6.82
```

For this file, select **HSQC** under **Two-column spectrum interpretation**. Select **HMBC** for an HMBC-only list, or **HSQC-TOCSY** for that experiment.

The default **Reject as ambiguous** prevents an unlabeled file from silently receiving the wrong experiment identity. Explicit row or section labels take precedence over the fallback choice.

**Do not combine unlabeled HSQC and HMBC peaks and assign one experiment to the whole file.** Label the rows or sections before uploading.

### Common format checks

| Check | Correct interpretation |
| --- | --- |
| First coordinate | δ¹³C in ppm |
| Second coordinate | δ¹H in ppm |
| Third column, if present | Experiment identity |
| Header | A descriptive header is accepted; numeric rows still determine the data |
| Extra columns | Do not rely on them being interpreted as metadata |
| Rejected nonnumeric rows | May be skipped; confirm the loaded peak count in the run log |

Experiment identity is essential because simulated HSQC correlations must be compared with experimental HSQC peaks, and likewise for HMBC and HSQC-TOCSY.

## 6. Configure the analysis

### Solvent

The supplied interface offers:

- Methanol-D4 (`CD3OD`).
- Chloroform-D1 (`CDCl3`).
- Dimethylsulphoxide-D6 (`DMSO-D6`).
- Unreported.

Choose the solvent used for the experimental acquisition. If it is not represented, use **Unreported** and document the actual solvent. This does not remove solvent-related prediction limitations.

### Clustering tolerances

| Control | Default | Function |
| --- | --- | --- |
| 13C tolerance (ppm) | 0.2 | Groups experimental carbon coordinates when building the peak graph |
| 1H tolerance (ppm) | 0.02 | Groups experimental proton coordinates when building the peak graph |
| Louvain/RBER resolution | 0.2 | Controls community detection on the experimental graph |

The two ppm tolerances are used in **experimental clustering**. They are not independent prediction acceptance windows of ±0.2 ppm for carbon and ±0.02 ppm for proton. The legacy matching calculation uses a separate weighted coordinate cost, described in Section 9.

Larger tolerances can connect more peaks and may merge otherwise distinct groups. Smaller tolerances can fragment groups. A higher community resolution generally favors smaller communities, but inspect the actual outcome rather than assuming a fixed cluster count.

Start with the defaults. If you explore alternatives, change one parameter at a time and assess whether the grouping and candidate interpretation remain reasonable. Do not select settings solely because they promote a preferred candidate.

### Experiment options

- **Use HMBC:** enabled by default.
- **Use HSQC-TOCSY:** disabled by default; enable when you have the corresponding data.
- HSQC is part of the standard workflow.

Keep the options consistent with the uploaded evidence. A missing experiment can be displayed as **N/A** or **Not evaluated**. That display prevents a missing measurement from being mistaken for a measured zero-match result; it does not establish that the legacy aggregate distance has been recalibrated for the missing experiment. Inspect single-experiment analyses with that limitation in mind.

### Use 2 HOSE spheres

This advanced legacy simulation option requests a less extensive local environment for HOSE-based prediction. The original documentation describes a two-sphere mode instead of the default three-sphere mode.

It is distinct from the **HOSE diagnostic threshold**, which filters reported support values after prediction. Changing the diagnostic threshold does not rerun the simulator.

In the inspected atomic exporter, the prediction call is made separately and does not read the `dotwobonds` option. Therefore, do not assume that enabling this option changes the complete atomic export in exactly the same way as the legacy 2D simulation. For a baseline analysis comparing atomic and correlation outputs, leave it disabled.

### Prediction engine

The Streamlit wrapper uses the HOSE predictor. The presence of `respredict` files in a distribution does not mean that this interface is running deep-learning predictions; deep-learning mode is disabled in the documented wrapper.

### Plot and diagnostic options

| Control | Purpose |
| --- | --- |
| Label simulated spectra | Adds legacy simulated-spectrum labels; dense labeling can be slow |
| Generate legacy PNG candidate plots | Produces optional static plots; disabled by default |
| Generate interactive HTML plots | Produces interactive Plotly spectra; enabled by default |
| Interactive plots — Top N candidates | Limits plot generation, not the size of the candidate list or atomic export |
| Plot appearance | Sets marker opacity for the next generated interactive plots |
| Debug output | Adds diagnostic information to the run output |

The default top-N value is **10**. Lower it if rendering is slow. Changing opacity alters presentation and does not alter peak matching or ranking.

## 7. Run the pipeline

Click **Run NMRfilter** after completing the sidebar configuration. The status panel shows the main stages:

1. **Preparing project:** creates the analysis workspace and parameter files.
2. **Converting candidate structures:** converts the structure list and, in the updated version, exports complete atomic predictions and simulator structures.
3. **Simulating candidate spectra:** generates the legacy predicted correlation records.
4. **Clustering measured peaks and ranking candidates:** builds the experimental graph, detects communities, matches candidate correlations, and generates results.

Runtime depends on the number of candidates, simulated correlations, experimental peaks, graph density, plotting settings, and available resources. A dense peak graph may take substantially longer than a small curated example.

During execution:

- Keep the page open.
- Read the status output before assuming a stage has stalled.
- Avoid starting several copies of the same analysis.
- If the pipeline stops, open **Run log** and identify the last completed stage.

Atomic predictions are generated before ranking. In the updated interface, **Calculated NMR data** may remain downloadable after a later-stage failure if the atomic output was successfully created. Such a download does not mean that ranking completed.

## 8. Understand experimental clustering

A graph vertex represents an experimental 2D peak. Relationships among similar carbon or proton coordinates connect vertices. Community detection then divides the graph into groups of correlated peaks.

These groups are **experimental peak communities**. If your input contains only HMBC data, they contain HMBC correlations. If you supplied HSQC and HMBC together, communities can contain peaks from both experiments.

**A cluster is not automatically a compound, a spin system, or an independently validated structural fragment.** Overlap and similar chemical shifts can connect signals from different constituents. Conversely, weak or missing peaks can split evidence from one molecule.

The current clustering code preserves legacy first-peak-anchored carbon and proton grouping, followed by merging carbon groups bridged by proton groups. Input order can therefore affect bin formation. Keep the exact peak list and row order for reproducibility.

The sidebar retains the name **Louvain/RBER resolution**. In the updated dependency implementation, community detection uses `leidenalg` with an RBER partition. When writing a methods section, distinguish the interface label from the algorithm actually used by your software version.

### Why a very large cluster matters

A large cluster can arise from genuinely dense evidence, overlap, artifacts, duplicates, or permissive tolerances. If most experimental peaks lie in one community, the cluster-distribution component of the ranking may have little discriminatory value.

Inspect the peak count, largest group sizes, and graph edge count in the log. Review the input before interpreting a large component as evidence for one dominant compound.

## 9. Interpret the ranking

Read **distance, standard deviation, and experiment-specific matching rates together**.

| Output | Interpretation |
| --- | --- |
| Rank | Relative candidate position under the implemented combined score |
| Distance | Normalized mismatch cost between predicted and experimental peak coordinates; lower is favored |
| Standard deviation | Normalized variation of experimental match fractions among peak communities; higher is favored by the combined ranking |
| Matching rate | Accepted matched peaks divided by the candidate's simulated correlations for the stated experiment |

### Distance

The legacy matching calculation assigns predicted correlations to experimental peaks using an assignment algorithm. For compatible experiment types, the coordinate cost is:

```text
weighted_difference = abs(experimental_13C - predicted_13C)
                    + 10 × abs(experimental_1H - predicted_1H)

pair_cost = weighted_difference²
```

Incompatible experiment types receive a very large cost. The assignment is global within the compared arrays, rather than an independent nearest-neighbor match for every peak. The raw candidate cost is the sum of assigned pair costs divided by the number of simulated correlations. An assigned peak counts as an accepted hit when its cost is **less than 9**.

The displayed distance is subsequently scaled relative to the candidate costs in that run.

**Distance = 0.00 does not mean identical chemical shifts.** It can identify the minimum-cost candidate after normalization, and the displayed number is rounded. A low distance with zero accepted matches does not support an identification.

Distance is not a shift error in ppm. Because normalization depends on the candidate list, values from separate runs with different candidates are not directly comparable as absolute fit measures.

### Standard deviation

For each experimental community, the program calculates:

```text
community_match_fraction = accepted matched experimental peaks in the community
                           / total experimental peaks in the community
```

It calculates the standard deviation of these fractions and normalizes that value across candidates. A higher value means that matches are distributed unevenly among communities. The ranking favors this concentration because a candidate in a mixture may match a subset of the experimental groups.

This value:

- Does not represent chemical shift uncertainty in ppm.
- Does not describe variation among replicate spectra.
- Does not provide a confidence interval or probability of identity.
- Can be **N/A** when the normalization has no variation to work with.

Thus, the clusters used here are not exclusively HMBC clusters unless the measured input is exclusively HMBC.

### Combined ranking

When both components have a usable range, the legacy combined score is:

```text
combined_score = [normalized_distance + (1 - normalized_standard_deviation)] / 2
```

Lower combined scores rank first. When the standard deviation cannot be normalized, the implementation uses a fallback based on normalized distance and a constant term. Experiment-specific matching rates are important supporting outputs; they are not independent identification probabilities.

### Matching rate

For each available experiment:

```text
matching_rate = accepted matched peaks / simulated candidate correlations
```

**5/6 means that five of six simulated candidate correlations were matched.** It does not mean that five of six experimental peaks were explained.

Example interpretation:

| Illustrative output | Meaning |
| --- | --- |
| HSQC: 5/6, 83.3% | Five accepted matches against six simulated HSQC correlations |
| HMBC: 8/20, 40.0% | Eight accepted matches against twenty simulated HMBC correlations |
| HMBC: N/A | No experimental HMBC evidence was supplied for evaluation |
| HMBC: 0/20 | HMBC evidence was supplied, but no accepted match was reported against twenty simulated correlations |

Always consider the denominator. A candidate with 2/2 matches has less supporting correlation evidence than one with 20/22, despite the higher percentage.

Simulated correlations are not necessarily unique experimentally resolved resonances. Equivalence and overlap can complicate the relationship between counts and the visible number of peaks.

## 10. Read the interactive spectra

Expand a candidate under **Interactive candidate plots**. HMBC and HSQC appear side by side when enabled; HSQC-TOCSY can add another panel. Carbon axes are aligned to help compare regions.

The horizontal axis is **δ¹H**, and the vertical axis is **δ¹³C**. Both follow the usual reversed NMR display direction.

### Marker legend

| Marker | Meaning |
| --- | --- |
| Solid green circle — Matched | An experimental peak accepted as a match for this candidate |
| Gray open circle — Simulated | A predicted candidate correlation; this trace includes predicted correlations whether or not they matched |
| Red open square — Unmatched | An experimental peak selected in the assignment but failing the hit cost criterion |
| Gray open square — Unused | An experimental peak outside the candidate's matched and assigned-unmatched arrays |

**Unused does not mean noise.** In a mixture, a real peak from another constituent may be unused for the candidate currently displayed. An unmatched peak is not automatically evidence of an incorrect structure; prediction limitations, experimental conditions, overlap, and incomplete data need consideration.

### Interactive inspection

- Hover over a marker to read coordinates, experiment, and marker category.
- Zoom into an aromatic, olefinic, oxygenated, or aliphatic region.
- Use the Plotly controls to pan, reset the view, and export the figure.
- Click legend entries to hide or show traces.
- Download the standalone **HTML** for inspection outside the app.

The initial view emphasizes approximately 0–10 ppm for proton and 0–200 ppm for carbon. Signals outside these displayed ranges require an adjusted view; being outside the initial axes does not mean they are absent from the underlying data.

The interactive plot uses the same matched and unmatched arrays as the ranking. A green point is therefore a visualization of the algorithm's decision, not a second independent validation.

### Opacity controls

The defaults emphasize accepted matches while keeping the surrounding evidence visible:

| Category | Default opacity |
| --- | --- |
| Matched | 0.95 |
| Simulated | 0.45 |
| Unmatched | 0.28 |
| Miscellaneous / unused | 0.12 |

Changing the sliders sets values for the next generated plots. It does not rewrite an HTML file you already downloaded. Increase the unused opacity when reviewing broader mixture evidence.

## 11. Explore structures and atomic shifts

Open **Structures and atomic shifts** after running the updated version.

1. Select a candidate from **Compound**. The list follows input candidate IDs and includes entry status.
2. Choose **13C** or **1H** under **Label shifts**.
3. Toggle **Show explicit hydrogens** as needed. Hydrogen labels are easier to inspect with explicit hydrogens visible.
4. Toggle **Show simulator atom indices**.
5. Set **Flag HOSE spheres below**.
6. Select a row in the atomic table to highlight that atom on the structure.

### Structure colors

| Color | Meaning |
| --- | --- |
| Amber | A usable prediction with HOSE sphere count below the selected threshold |
| Red | No usable prediction or no usable HOSE support value |
| Blue | The atom selected in the table; selection can override its diagnostic color |

Labels show simulated chemical shifts in ppm. The drawing may round labels to two decimals; use the exported table for the stored value.

### Atom indices and safe mapping

The atom indices belong to the simulator after its molecular preparation and explicit-hydrogen expansion. They are **one-based internal indices**, not conventional chemical numbering or guaranteed indices from an external SMILES parser.

The structure viewer uses a saved simulator molecule and an atom map, with checks on atom identity, elements, and coordinates. Hiding hydrogens is a display operation; it does not redefine the original indices.

Do not reconstruct assignments by pasting the SMILES into another program and assuming its atom number 7 is the simulator's atom number 7. Use the exported simulator molecule and atom map.

### Structure downloads

- **Download annotated structure (.svg):** a scalable drawing with the currently selected labels and highlights.
- **Download simulator structure (.mol):** the saved molecular representation used for mapping.
- **Download atom map (.csv):** candidate identity, simulator atom index, atom ID, element, and coordinates.

Earlier results without saved simulator structures cannot be mapped safely from SMILES alone. Run the updated version again to generate the required mapping files.

## 12. Download and navigate the outputs

### Download buttons

| Button | Contents and purpose |
| --- | --- |
| Download ranking table (.tsv) | Candidate ranking, displayed metrics, and experiment-specific matching results |
| Download HTML | One candidate's interactive spectral comparison |
| Download all interactive plots (.zip) | The generated top-N HTML plots |
| Download calculated NMR data (.zip) | All input entries, complete atomic predictions, organized correlations, entry status, and original prediction outputs |
| Download complete project/results ZIP | The analysis workspace, numerical results, generated plots, and calculated NMR export material |
| Download compound diagnostic (.csv) | The current HOSE compound summary |
| Download flagged atoms (.csv) | Atoms flagged under the current nucleus filter and threshold |

The **top-N plotting limit does not restrict the atomic export to top-ranked candidates**.

### Files inside the calculated NMR ZIP

| File or location | What it contains |
| --- | --- |
| `resultprediction.csv` | In the updated calculated-data ZIP, the complete atomic prediction table with headers and candidate identity |
| `atomic_predictions.csv` | The same complete atomic data with an explicit filename |
| `atomic_prediction_status.csv` | One status record for every nonempty input entry |
| `entries_without_prediction.csv` | Entries whose status is not fully `predicted`, including partial predictions and errors |
| `calculated_correlations.csv` | Simulated 2D correlations with English column names |
| `organized_correlations.csv` | Organized 2D correlations with descriptive Portuguese headers retained for compatibility |
| `candidate_index.csv` | Candidate identities, experiment-specific correlation counts, and atomic status |
| `candidates/` | A numbered folder per candidate, including atomic and experiment-specific tables |
| `simulator_structures/` | Saved `.mol` structures and atom maps, when successfully generated |
| `original/` | Unmodified simulator outputs and effective parameters |
| `README.txt` | Export-specific descriptions and completeness information |

Each candidate folder contains `atomic_predictions.csv`, `13C.csv`, `1H.csv`, `HSQC.csv`, `HMBC.csv`, `HSQCTOCSY.csv`, and `structure.smi`.

### The important `resultprediction.csv` distinction

The raw legacy simulator file and the organized export use the same basename in different locations:

- **Top-level `resultprediction.csv` in the updated calculated-data ZIP:** complete atomic predictions.
- **`original/resultprediction.csv`:** raw legacy 2D correlation records, including experiment codes and candidate separators.
- **`calculated_correlations.csv`:** the organized 2D correlation table with English headers.

Older export versions used the top-level filename for organized correlations. Check the ZIP's `README.txt` and column headers before assuming the meaning of the file.

In the complete project ZIP, calculated-data files are also grouped under `calculated_NMR/`. The application's working `result/resultprediction.csv` remains the legacy simulation output; do not confuse it with `calculated_NMR/resultprediction.csv`.

### Internal experiment codes

| Legacy code | Experiment |
| --- | --- |
| `q` | HSQC |
| `b` | HMBC |
| `t` | HSQC-TOCSY |

Organized correlation tables expand these codes into readable experiment names.

The atomic and correlation exports preserve **input order**, not ranking order. Join tables by candidate identity and verify the SMILES; do not join them by displayed row position alone.

## 13. Understand the atomic prediction tables

Complete atomic predictions are obtained directly from the HOSE prediction tool. They are not reconstructed from HMBC or HSQC lists.

This matters because a carbon without an attached hydrogen will not have a direct HSQC correlation. A 2D correlation list alone cannot provide a complete independent table of all predicted carbon and hydrogen atoms.

### Atomic fields

| Column | Meaning |
| --- | --- |
| `candidate_id` | One-based candidate identifier in the nonempty input-entry sequence |
| `input_line` | Original line number in the structure file |
| `candidate_name` | Resolved candidate name |
| `smiles` | Input structure representation |
| `nucleus` | `13C` or `1H` |
| `atom_index` | One-based simulator index after molecular preparation and explicit-hydrogen expansion |
| `atom_id` | Simulator/CDK atom identifier |
| `shift_ppm` | Predicted mean chemical shift reported by the predictor |
| `minimum_ppm` | Lower value returned by the predictor |
| `maximum_ppm` | Upper value returned by the predictor |
| `hose_spheres` | Local-environment sphere count reported for the database match |
| `prediction_status` | Whether the atom was predicted, had no prediction, or encountered a prediction error |
| `message` | Diagnostic explanation, where applicable |
| `prediction_source` | Predictor provenance |

The minimum and maximum values are predictor outputs. They should not be presented as validated statistical confidence intervals or experimental uncertainty.

Equivalent atoms are not collapsed in the complete atomic table. Three equivalent methyl hydrogens can therefore produce three atomic rows even if their predicted shifts are identical. Atomic row counts do not equal numbers of resolved resonances or 2D peaks.

### Missing values and entry status

An unavailable chemical shift is **blank**, rather than zero or a failure sentinel. A real numerical zero must not be used as a substitute for missing data.

| Entry status | Interpretation |
| --- | --- |
| `predicted` | All exported target atoms received usable predictions |
| `partial_prediction` | Some target atoms were predicted and others were unavailable or failed |
| `no_prediction` | No target atom received a usable prediction |
| `no_target_atoms` | No carbon or hydrogen target atoms were found |
| `structure_error` | Structure parsing or preparation failed |

Every nonempty input entry has a status record. A structure error may have no atomic rows, so use `atomic_prediction_status.csv` to assess completeness rather than counting only rows in `atomic_predictions.csv`.

If the legacy correlation blocks cannot be mapped unambiguously to the input entries, the exporter preserves the raw output and avoids attaching partial blocks to incorrect candidates. Inspect `README.txt` for the **Legacy correlations complete** flag. An empty organized correlation table in that situation does not necessarily mean that no raw correlations were generated.

## 14. Use HOSE diagnostics

Expand **HOSE diagnostics — compounds with limited database support** in the analysis tab.

HOSE codes describe successive layers of an atom's local molecular environment. A match using fewer spheres is less structurally specific than a match using more layers. This provides a way to flag predictions that deserve closer inspection.

### Controls

1. Select **Flag predictions with fewer than this number of HOSE spheres**.
2. Choose **Both**, **13C**, or **1H** under **Inspect nucleus**.
3. Review the compound summary.
4. Inspect the flagged atomic rows.
5. Download the compound and atom diagnostic CSVs.

The default threshold is **4**, which flags usable predictions with sphere counts **1–3**. Missing or unusable predictions are also included in the diagnostic.

The threshold is a configurable screening choice. It is not a validated cutoff for accurate versus inaccurate prediction, and it does not change the ranking or rerun predictions.

### Compound summary fields

| Field | Meaning |
| --- | --- |
| `minimum_hose_spheres` | Lowest usable support count among atoms included by the selected nucleus filter |
| `low_13C_atoms` | Number of carbon predictions below the selected threshold |
| `low_1H_atoms` | Number of hydrogen predictions below the selected threshold |
| `atoms_without_hose` | Atoms without usable prediction/support information |
| `total_target_atoms` | Number of atomic rows included for that candidate under the nucleus filter |

Counts refer to individual atoms, including equivalent hydrogens. Compare both the number and proportion of flagged atoms when assessing compounds of different sizes.

### What low support tells you

Low support suggests that the bundled predictor found only a less specific local-environment match. It does not establish that:

- The candidate structure is incorrect.
- The predicted shift must be inaccurate.
- Experimental data are absent from the current online nmrshiftdb2 database.
- A highly supported prediction is sufficient to confirm compound identity.

Use the flag to direct literature review, experimental assignment, and inspection of the relevant atom on the structure.

### Which database files are used?

The inspected distribution bundles the reference tables **`nmrshiftdbc.csv`** and **`nmrshiftdbh.csv`** inside **`lib/simulate.jar`**, for carbon and hydrogen prediction, respectively.

These are bundled reference resources. They are distinct from `atomic_predictions.csv`, which contains predictions for your submitted structures. Updating or contributing to the live online database does not automatically replace the resources inside the deployed JAR.

## 15. Contribute experimental assignments to nmrshiftdb2

The diagnostic can identify compounds for which better experimental reference data would be useful. The application links to [nmrshiftdb2](https://nmrshiftdb.nmr.uni-koeln.de/) and its [submission and review instructions](https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/using.html).

A practical route is:

1. Identify the flagged compound and atoms.
2. Look for reliable experimental assignments in your own measurements or the literature.
3. Verify the structure and assignments with suitable evidence.
4. Map assignments carefully to the submitted structure; simulator indices are not conventional chemical numbering.
5. Include solvent, relevant acquisition conditions, and source references.
6. Follow the database's current submission and review process. The interface describes public availability after reviewer approval.

**Do not submit simulated shifts as experimental measurements.** An automated match in an unresolved mixture is also insufficient on its own to establish a complete experimental assignment.

The application uses a database snapshot. A contribution becomes relevant to the local predictor only when the corresponding reference resources are updated in the software deployment.

## 16. Build confidence in a candidate

For each leading candidate, ask:

1. Is the structure chemically plausible for the sample?
2. Are the supplied name and SMILES consistent?
3. Are there enough accepted correlations to support a meaningful interpretation?
4. Does HSQC agree with the expected protonated carbon environments?
5. Does HMBC provide discriminating connectivity evidence?
6. Are important predicted regions unsupported, and can that be explained experimentally?
7. Are matches concentrated in a coherent subset of experimental communities?
8. Do flagged HOSE atoms coincide with questionable or decisive predictions?
9. Do closely related alternatives explain the same evidence?
10. Is the interpretation stable under reasonable changes to analysis settings?

### Appropriate conclusions

| Observation | Defensible interpretation |
| --- | --- |
| High rank with coherent HSQC/HMBC evidence | Candidate merits targeted investigation |
| High percentage based on very few correlations | Compatible but limited evidence |
| Similar results for closely related structures | Candidate distinction remains unresolved |
| Low distance with no accepted matches | Relative cost alone does not provide supporting identification evidence |
| Missing HMBC acquisition | Long-range connectivity was not evaluated |
| Many low-HOSE atoms | Prediction support needs closer review |

Follow-up may include improved peak assignment, additional NMR experiments, comparison with authentic material, fractionation, and integration with independent MS or other analytical evidence.

NMRfilter matching rates are not abundance estimates. They do not quantify a compound's concentration or its fraction of the mixture.

## 17. Troubleshooting

### No readable 13C/1H peak pairs

Check that the first two columns are numeric coordinates, that they are in carbon–proton order, and that the delimiter is consistent. Remove leading peak IDs, units appended to numbers, and spreadsheet formatting. Confirm that a header is followed by actual numeric rows.

### Peaks have no experiment type

For a single-experiment two-column list, select its experiment under **Two-column spectrum interpretation**. For mixed data, add section labels or a third `type` column. Changing the experiment checkbox does not label an ambiguous input file.

### All candidates have zero matches

First inspect experiment labels, axis order, units, and chemical shift referencing. Then confirm that the expected numbers of measured peaks and candidate spectra were loaded. Check that the input is actually the intended spectrum and that the candidate structures are appropriate.

Do not infer success from a distance of 0.00. Download the project files and review the raw predictions, measured input, and log if the issue persists.

### Candidate names appear wrong, or ranking raises an index error

Compare the structure and name lists line by line. Remove blank lines and check that the names count and order match the candidates. Confirm that conversion did not fail for an entry. Do not repair a mislabeled result by relabeling rank positions without checking structure identity.

### A stage seems stalled

Open the live status and run log. Check whether the program is clustering a dense graph, matching a large candidate set, or generating labels and plots.

Try a smaller curated candidate set and peak list. Disable optional PNG plots and simulated labels, and reduce the number of HTML plots. Changes to clustering tolerances should be scientifically justified rather than used only to force a faster run.

### Bruker background image unavailable

The peak-list workflow intentionally does not supply raw Bruker background data. A warning that an HMBC or HSQC Bruker path is unconfigured can concern the background image rather than the numerical matching. Inspect the final stage status to determine whether ranking completed.

### Ranking succeeds but interactive plots are missing

Confirm that **Generate interactive HTML plots** was enabled before the run. Inspect the log and the complete project ZIP for `plots_html/`. A plotting or embedding error can occur after numerical ranking has completed.

### Structure tab asks for a new run

Older results lack the persisted simulator molecule and atom map. Run the updated application once. The viewer intentionally avoids guessing atom assignments from a fresh SMILES drawing.

### Atomic export contains blanks

Read `prediction_status`, `message`, and the entry status table. Blank shifts indicate unavailable predictions, not zero ppm. Use HOSE diagnostics to locate the affected atoms.

### Atomic export is present but correlations are empty

Read the export `README.txt` and entry statuses. The atomic and correlation workflows are separate. An incomplete legacy correlation stream may be preserved only as raw output to prevent incorrect candidate attribution.

### Download does not open

Wait for completion and download again. Confirm that a ZIP has a nonzero size and opens with an archive utility. For the full atomic data, use **Download calculated NMR data (.zip)** rather than the ranking TSV or plots ZIP. Open a downloaded HTML file in a browser.

### Results disappear after a page interaction

Streamlit reruns the interface when controls change. The newer atomic download, diagnostic, and structure sections retain a project reference in session state, but some ranking and plot displays are created within the run action. Save the outputs promptly; if needed, rerun with the documented settings.

### What to send when reporting a problem

Provide the error text, last completed stage, parameter values, input counts, and a reproducible example. Relevant diagnostic files include:

- `realspectrum.csv` and `testall.smi`.
- `testallnames.txt`, if used.
- `nmrproc.properties`.
- Raw `resultprediction.csv`.
- `atomic_predictions.csv` and `atomic_prediction_status.csv`.
- `cluster.txt`, `clusterslouvain.txt`, and relevant `smart*.csv` files.
- `result.txt`, `ranking_table.tsv`, and the run log.

A small known positive example is particularly useful for distinguishing parsing, prediction, clustering, matching, and rendering problems.

## 18. Reproducibility and reporting

Archive:

- The exact candidate list and names in input order.
- The measured peak file in its original row order.
- Acquisition solvent, experiment identities, and chemical shift referencing information.
- Software version or repository commit, when available, and analysis date.
- All effective parameters, including experiment options and HOSE mode.
- Complete project/results ZIP, ranking TSV, atomic exports, and plots used in interpretation.
- HOSE thresholds and nucleus filters used for exported diagnostic summaries.

Community detection is not explicitly given a fixed seed in the inspected implementation. Avoid claiming bit-for-bit reproducibility across reruns or software environments without verifying it.

### Suggested methods wording

Replace the bracketed fields with your actual settings:

> Candidate structures supplied as SMILES were evaluated against experimental [HSQC/HMBC/HSQC-TOCSY] peak lists using the NMRfilter Streamlit interface [version or commit; access date]. Chemical shifts were predicted using the bundled HOSE reference database with [solvent setting]. Experimental peak grouping used carbon and proton tolerances of [value] and [value] ppm, respectively, followed by RBER community detection using the implementation provided in that version with resolution [value]. Candidate ranking combined normalized assignment cost and normalized variation of experimental match fractions among communities. Experiment-specific matching rates were expressed relative to simulated candidate correlation counts. Ranked candidates were reviewed through spectral overlays and atomic prediction diagnostics and treated as hypotheses for further structural assessment.

### Suggested results wording

> Candidate X was prioritized by NMRfilter and showed [matched/total] HSQC and [matched/total] HMBC correlations. Inspection of the matched regions supported compatibility with [specified structural evidence]. [Number] atomic predictions fell below the selected HOSE support threshold. These findings support further investigation of the candidate; they do not establish an unambiguous identification.

Report missing experiments explicitly. Do not describe N/A as a 0% matching rate or the displayed standard deviation as chemical shift uncertainty.

## 19. Local execution and deployment notes

These notes are for users maintaining a local copy or the online deployment. They are not required to use the hosted application.

### Windows local execution

The distributed Windows package includes a launcher:

1. Extract the ZIP completely.
2. Run `START_NMRFILTER.bat`.
3. On first use, the launcher prepares the `nmrfilter` Conda environment, with the Python and Java versions specified by that package, installs dependencies, and starts Streamlit.
4. Reuse the launcher for subsequent runs.

The documented launcher configuration uses Python 3.11 and Java 17. Avoid launching with an unrelated base environment or a globally installed Streamlit executable. If dependencies become inconsistent, the package may include `RESET_NMRFILTER_ENV.bat` to rebuild the dedicated environment.

### Streamlit Community Cloud

The main application file is `app.py`. The repository's requirements and system-package configuration provide the Python and Java dependencies. Java, RDKit, SciPy, igraph, and Leiden-related dependencies need to be available for the corresponding pipeline stages.

Do not retain machine-specific Windows paths in a Linux deployment. Java properties treat backslashes as escape characters; the wrapper writes forward-slash paths to avoid malformed Windows paths.

Keep generated project folders, temporary simulation directories, caches, and logs out of source-control packages. Including previous analysis projects can substantially inflate a distribution ZIP and mix generated outputs with application code. The downloadable **complete results ZIP** should archive an analysis project; the **application ZIP** should distribute the software.

The presence of large optional prediction resources does not imply that they are used by the hosted HOSE workflow. Check actual file sizes against current hosting and repository limits when deploying.

## 20. Frequently asked questions

### Is NMRfilter identifying a compound automatically?

It ranks the supplied candidates by compatibility with NMR evidence. Confirmation requires additional structural assessment.

### Are the experimental clusters HMBC correlations?

They are groups of experimental peaks. They contain only HMBC correlations if the input contains only HMBC. Mixed HSQC/HMBC input can produce communities containing both types.

### Does 0.00 distance mean an exact match?

No. It is a normalized, rounded value relative to the candidate set. Check accepted matches and the actual coordinates.

### Does a higher standard deviation mean worse prediction precision?

No. It describes variation of match fractions among experimental communities. The combined ranking favors higher normalized values.

### Does 5/6 mean five experimental peaks out of six were explained?

It means five accepted matches against six simulated candidate correlations for the stated experiment.

### Can I use only HSQC or only HMBC?

The parser accepts explicitly labeled single-experiment data. Missing experiments should be reported as not evaluated. The legacy aggregate cost must still be interpreted cautiously; the display of N/A alone does not validate single-experiment ranking behavior. Combined, well-curated HSQC and HMBC evidence is more informative for structural assessment.

### Are all candidate predictions exported, or only those shown in plots?

The updated calculated NMR export includes every nonempty input entry in the status table and all available atomic rows. Plot generation is limited separately by top N.

### Which file contains the complete predicted ¹³C and ¹H data?

Use `atomic_predictions.csv`, or the top-level `resultprediction.csv` in the updated calculated-data ZIP. Use `atomic_prediction_status.csv` to check failed or partial entries.

### Are the structure labels experimental assignments?

No. They are simulated atomic chemical shifts. The green markers in spectral plots show accepted experimental correlation matches; these are a different type of output.

### Are the displayed atom numbers standard chemical numbering?

No. They are simulator indices. Use the saved molecule and atom map to preserve identity.

### Does low HOSE support prove that nmrshiftdb2 lacks the compound?

No. It describes matching against the bundled database snapshot. The current online database may have additional data.

### Does the HOSE threshold change the ranking?

No. It changes which predictions are flagged in the diagnostic and structure view. It is separate from the simulation's two-sphere option.

### Can I compare normalized distances across different candidate lists?

Not as absolute fit measures. Adding or removing candidates changes the normalization range.

### Can I quantify mixture composition from matching rates?

No. Matching rates describe correlation compatibility and are not calibrated abundance measures.

## 21. References and contact

### Cite the original NMRfilter methodology

Kuhn, S., Colreavy-Donnelly, S., de Andrade Silva Quaresma, L. E., et al. (2020). Applying NMR compound identification using NMRfilter to match predicted to experimental data. *Metabolomics*, **16**, 123. [https://doi.org/10.1007/s11306-020-01748-1](https://doi.org/10.1007/s11306-020-01748-1).

### Related integrated MS/NMR mixture analysis

Kuhn, S., Colreavy-Donnelly, S., de Souza, J. S., and Borges, R. M. (2019). An integrated approach for mixture analysis using MS and NMR techniques. *Faraday Discussions*, **218**, 339–353.

When using the Streamlit adaptation, also record the [repository](https://github.com/RicardoMBorges/NMRfilter_Streamlit), application access date, and software version or commit when available.

### Contact

- **Ricardo M Borges:** [ricardo_mborges@ufrj.br](mailto:ricardo_mborges@ufrj.br)
- **Stefan Kuhn:** [stefan.kuhn@ut.ee](mailto:stefan.kuhn@ut.ee)

For technical questions, include the error text, relevant parameters, input counts, and a small reproducible example whenever possible.
