# Complete atomic NMR export

The app now runs `uk.ac.dmu.simulate.AtomicPredictions` before the existing conversion and 2D simulation stages. The compiled extension is `lib/atomic-export.jar`; source is in `java_src/uk/ac/dmu/simulate/AtomicPredictions.java`.

It calls the original `PredictionTool.predict(molecule, atom, true, solvent)` for EVERY carbon and hydrogen. Preparation follows Simulate: read with the CDK SMILES reader, generate coordinates, expand hydrogens using `Simulate.addAndPlaceHydrogens`, assign atom IDs and detect aromaticity. The original simulator JAR and databases are unchanged. The app uses HOSE prediction; deep-learning export is explicitly rejected rather than silently substituting a different model.

The NMR download contains:

- `resultprediction.csv` and `atomic_predictions.csv`: complete atom table, including target atoms without a prediction.
- `atomic_prediction_status.csv`: every nonempty input entry, in input order, with counts and status.
- `entries_without_prediction.csv`: structures with missing/partial predictions, errors or no C/H atoms.
- Per-structure `13C.csv`, `1H.csv`, `atomic_predictions.csv`, HSQC/HMBC/HSQCTOCSY tables and SMILES.
- `calculated_correlations.csv` and `organized_correlations.csv`: existing organized 2D outputs.
- `original/`: raw prediction records and run properties.

`atom_index` is one-based in the explicit-hydrogen simulator molecule, consistent with legacy correlations. `atom_id` is the CDK identifier. Do not interpret these indices as the numbering produced by another SMILES parser. Equivalent atoms remain separate. Blank `shift_ppm` means no prediction; the simulator's failure sentinel is never represented as a shift. HOSE sphere count, minimum and maximum are retained.

Names come from the optional names file, then inline SMI names, then candidate_N. `input_line` retains the original line number. Duplicate structures remain separate entries. Atomic download remains accessible if a later 2D/ranking stage fails. Incomplete legacy blocks remain in the raw output and are not assigned to structures.

Run `python -m unittest discover -s . -p test_atomic_export.py -v` to verify real HOSE predictions, carbonyl/OH coverage, exact equality with original 2D shifts, invalid/duplicate identities and raw preservation. Java and the bundled lib dependencies are required.

Rebuild the extension with a JDK:

    javac --release 8 -cp "lib/*" -d build_java java_src/uk/ac/dmu/simulate/AtomicPredictions.java
    jar cf lib/atomic-export.jar -C build_java .

## Package provenance

This update was prepared from the locally recoverable `NMRfilter_Streamlit_NMR_Export.zip` working tree from the previous export task. The later attachment `NMRfilter_Streamlit_NMR_Export(2).zip` could not be downloaded in this session, so differences between those two packages have not been checked. Existing files in the recovered package were preserved except app.py and prediction_export.py, which implement this export; new Java extension, source, documentation and tests were added.
