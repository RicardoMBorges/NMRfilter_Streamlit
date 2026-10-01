"""Draw simulator molecules using their persisted atom order and coordinates."""
from pathlib import Path
import math
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def load_simulator_molecule(directory, candidate_id):
    stem = f'candidate_{int(candidate_id):04d}'
    directory = Path(directory)
    mapping = pd.read_csv(directory / f'{stem}_atom_map.csv', keep_default_na=False)
    mol = Chem.MolFromMolFile(str(directory / f'{stem}.mol'), removeHs=False, sanitize=False)
    if mol is None or mol.GetNumAtoms() != len(mapping):
        raise ValueError('Saved simulator molecule and atom map differ; drawing stopped.')
    if mol.GetNumConformers() != 1:
        raise ValueError('Simulator drawing coordinates are unavailable.')
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    conformer = mol.GetConformer()
    for index, row in mapping.iterrows():
        atom = mol.GetAtomWithIdx(index)
        if int(row['atom_index']) != index + 1 or atom.GetSymbol() != row['element'] or int(row['candidate_id']) != int(candidate_id):
            raise ValueError('Simulator atom identity check failed; drawing stopped.')
        point = conformer.GetAtomPosition(index)
        if row['x'] == '' or row['y'] == '' or abs(point.x-float(row['x'])) > 0.001 or abs(point.y-float(row['y'])) > 0.001:
            raise ValueError('Saved coordinates differ from simulator atom map.')
        atom.SetIntProp('_simulator_index', index + 1)
        atom.SetProp('_simulator_id', str(row['atom_id']))
    return mol


def draw_simulator_structure(directory, candidate_id, rows, nucleus='13C', show_hydrogens=False, show_indices=True, threshold=4, selected=None):
    mol = load_simulator_molecule(directory, candidate_id)
    assignments = {int(r['atom_index']):r for _, r in rows.iterrows()}
    if len(assignments) != len(rows):
        raise ValueError('Duplicate prediction atom indices; drawing stopped.')
    for index, record in assignments.items():
        if not 1 <= index <= mol.GetNumAtoms():
            raise ValueError('Prediction atom index is outside the saved structure.')
        atom = mol.GetAtomWithIdx(index-1)
        expected = {'13C':'C', '1H':'H'}.get(record['nucleus'])
        if atom.GetSymbol() != expected or atom.GetProp('_simulator_id') != str(record['atom_id']):
            raise ValueError('Prediction atom ID/nucleus differs from simulator map.')
    # Removing H is a DISPLAY operation only. Keep the explicit simulator index
    # on every remaining atom; never assign predictions by the new RDKit index.
    if not show_hydrogens:
        editable = Chem.RWMol(mol)
        for i in reversed(range(editable.GetNumAtoms())):
            atom = editable.GetAtomWithIdx(i)
            if atom.GetSymbol() == 'H' and atom.GetIntProp('_simulator_index') != selected:
                editable.RemoveAtom(i)
        mol = editable.GetMol()
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
    highlights, colors = [], {}
    for atom in mol.GetAtoms():
        original = atom.GetIntProp('_simulator_index')
        record = assignments.get(original)
        note = f'#{original}' if show_indices else ''
        if record is not None and record['nucleus'] == nucleus:
            valid = record['prediction_status'] == 'predicted'
            value = pd.to_numeric(record['shift_ppm'], errors='coerce')
            note += (' ' if note else '') + (f'{value:.2f} ppm' if valid and pd.notna(value) else 'no prediction')
            spheres = pd.to_numeric(record['hose_spheres'], errors='coerce')
            if not valid or pd.isna(spheres) or spheres <= 0:
                highlights.append(atom.GetIdx()); colors[atom.GetIdx()] = (0.9,0.45,0.45)
            elif spheres < threshold:
                highlights.append(atom.GetIdx()); colors[atom.GetIdx()] = (1.0,0.78,0.3)
        if original == selected:
            if atom.GetIdx() not in highlights: highlights.append(atom.GetIdx())
            colors[atom.GetIdx()] = (0.35,0.7,1.0)
        if note: atom.SetProp('atomNote', note)
    width = 1100
    height = max(540, min(900, 400 + mol.GetNumAtoms()*5))
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.prepareMolsBeforeDrawing = False # reuse saved simulator coordinates
    options.padding = 0.12
    options.annotationFontScale = 0.7
    drawer.DrawMolecule(mol, highlightAtoms=highlights, highlightAtomColors=colors)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def render_structure_tab(project):
    import streamlit as st
    if project is None:
        st.info('Run NMRfilter to view atom assignments on each structure.')
        return
    project = Path(project)
    atom_file = project/'result/atomic_predictions.csv'
    status_file = project/'result/atomic_prediction_status.csv'
    directory = project/'result/simulator_structures'
    if not atom_file.exists() or not status_file.exists() or not directory.exists():
        st.info('Run this updated version once to save the simulator structures and atom maps. Earlier result files cannot be mapped safely from SMILES alone.')
        return
    try:
        statuses = pd.read_csv(status_file, keep_default_na=False)
        if statuses.empty:
            st.info('No input structures were recorded.'); return
        ids = statuses['candidate_id'].astype(int).tolist()
        labels = {int(r.candidate_id): f'{r.candidate_id} — {r.candidate_name} ({r.status})' for r in statuses.itertuples()}
        cid = st.selectbox('Compound', ids, format_func=labels.get, key='structure_candidate')
        metadata = statuses[statuses['candidate_id'].astype(int)==cid].iloc[0]
        st.code(str(metadata['smiles']), language=None)
        stem=f'candidate_{cid:04d}'
        if not (directory/f'{stem}.mol').exists():
            st.warning(f"No saved simulator structure: {metadata['status']}. {metadata['message']}")
            return
        atoms = pd.read_csv(atom_file, keep_default_na=False)
        atoms = atoms[atoms['candidate_id'].astype(int)==cid]
        nucleus = st.radio('Label shifts', ['13C','1H'], horizontal=True, key='structure_nucleus')
        controls = st.columns(3)
        show_h = controls[0].checkbox('Show explicit hydrogens', value=nucleus=='1H', key='structure_show_h_'+nucleus)
        show_indices = controls[1].checkbox('Show simulator atom indices', True, key='structure_indices')
        cutoff = controls[2].selectbox('Flag HOSE spheres below', [4,3,2,5,6], key='structure_hose_cutoff')
        visible = atoms[atoms['nucleus']==nucleus].reset_index(drop=True)
        st.caption('Select a row to highlight its atom in blue. Atom indices belong to the simulator; they are not conventional chemical numbering.')
        event = st.dataframe(visible, use_container_width=True, hide_index=True, on_select='rerun', selection_mode='single-row', key=f'structure_table_{cid}_{nucleus}')
        selected = int(visible.iloc[event.selection.rows[0]]['atom_index']) if event.selection.rows else None
        svg = draw_simulator_structure(directory, cid, atoms, nucleus, show_h, show_indices, int(cutoff), selected)
        st.image(svg, use_container_width=True)
        st.caption('Amber: low HOSE sphere count · Red: no usable prediction · Blue: selected atom. Shifts are simulated, in ppm. For crowded drawings, hide indices or hydrogens and select individual atoms in the table.')
        st.download_button('Download annotated structure (.svg)', svg.encode('utf-8'), file_name=f'{stem}_{nucleus}_annotated.svg', mime='image/svg+xml', key='structure_svg_download')
        st.download_button('Download simulator structure (.mol)', (directory/f'{stem}.mol').read_bytes(), file_name=f'{stem}.mol', mime='chemical/x-mdl-molfile', key='structure_mol_download')
        st.download_button('Download atom map (.csv)', (directory/f'{stem}_atom_map.csv').read_bytes(), file_name=f'{stem}_atom_map.csv', mime='text/csv', key='structure_map_download')
    except (OSError, ValueError, KeyError) as error:
        st.warning(f'Structure view unavailable: {error}')
