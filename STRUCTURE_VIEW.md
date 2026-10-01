# Structures and atomic shifts

The new tab uses the exact molecule exported by the atomic simulator, including explicit hydrogens, original atom order and the simulator's 2D coordinates. It does not regenerate the structure from SMILES.

Run the updated app again to create `result/simulator_structures/candidate_NNNN.mol` and the corresponding `_atom_map.csv`. Older atomic CSVs alone are insufficient for a safe drawing.

Select a compound, choose 13C or 1H, and select a table row to highlight its atom. Amber indicates predictions below the selected HOSE sphere threshold; red indicates no usable prediction; blue indicates the selected atom. Individual hydrogens remain separate. Controls allow hiding hydrogen atoms and simulator indices to reduce crowding. A selected hydrogen remains visible even when other hydrogens are hidden.

The drawing module verifies the MOL atom count, elements, coordinates, candidate ID and prediction atom IDs before attaching shifts. Display-only removal of hydrogens preserves the original simulator index on every remaining atom. Conventional chemical numbering is not inferred.

Annotated SVG, original MOL and atom mapping are downloadable from the tab. Saved structures are also included in the calculated NMR export and the full results export. Projects and caches are excluded from the app distribution ZIP.

Verification: real simulator export for ethanol/acetic acid; C/H drawings with and without explicit H; selection of a hidden hydrogen; rejected atom-ID mismatch; preserved MOL in prediction ZIP; existing direct atomic/2D agreement and invalid-entry identity tests. Example SVG was rendered and inspected. The full ranking workflow and interactive browser UI were not rerun in this environment.
