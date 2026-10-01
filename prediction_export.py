"""Export the original prediction records without recomputing NMR shifts."""
import configparser
import csv
import io
import re
import zipfile
from pathlib import Path

EXPERIMENTS = {'q': 'HSQC', 'b': 'HMBC', 't': 'HSQCTOCSY'}
FIELDS = ['candidate_id', 'candidate_name', 'smiles', 'experiment', 'carbon_ppm', 'proton_ppm', 'carbon_atom_index', 'hydrogen_atom_index']

def csv_bytes(rows, fields):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode('utf-8-sig')

def build_prediction_zip(project):
    project = Path(project)
    cp = configparser.ConfigParser()
    cp.read(project / 'nmrproc.properties')
    filename = cp.get('onesectiononly', 'predictionoutput', fallback='resultprediction.csv')
    source = project / 'result' / filename
    atom_source = project / 'result' / 'atomic_predictions.csv'
    status_source = project / 'result' / 'atomic_prediction_status.csv'
    if not atom_source.is_file() or not status_source.is_file():
        raise FileNotFoundError('Complete atomic predictions are unavailable. Run the updated simulator; 2D correlations cannot substitute for atomic predictions.')
    with atom_source.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f); atomic_fields = reader.fieldnames; atomic_rows = list(reader)
    with status_source.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f); status_fields = reader.fieldnames; status_rows = list(reader)
    smiles = [s.strip().split(maxsplit=1)[0] for s in (project / 'testall.smi').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if len(status_rows) != len(smiles) or any(int(r['candidate_id']) != i or r['smiles'] != smiles[i-1] for i,r in enumerate(status_rows,1)):
        raise ValueError('Atomic status/input identities differ; export stopped.')
    for r in atomic_rows:
        i = int(r['candidate_id'])
        if not 1 <= i <= len(smiles) or r['smiles'] != smiles[i-1] or r['candidate_name'] != status_rows[i-1]['candidate_name']:
            raise ValueError('Atomic row identity differs from input; export stopped.')
    names_path = project / 'testallnames.txt'
    names = names_path.read_text(encoding='utf-8-sig').splitlines() if names_path.exists() else []
    blocks, current = [], []
    for line_number, line in enumerate((source.read_text(encoding='utf-8-sig').splitlines() if source.exists() else []), 1):
        line = line.strip()
        if not line:
            continue
        if line == '/':
            blocks.append(current)
            current = []
            continue
        parts = next(csv.reader([line]))
        if len(parts) != 5 or parts[2] not in EXPERIMENTS:
            raise ValueError(f'Unrecognized prediction record at line {line_number}.')
        float(parts[0]); float(parts[1]); int(parts[3]); int(parts[4])
        current.append(parts)
    if current:
        blocks.append(current)
    correlations_complete = len(blocks) == len(smiles)
    if not correlations_complete:
        blocks = [[] for _ in smiles]  # Incomplete legacy output must never shift candidate identities.
    all_rows, index = [], []
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as archive:
        for i, block in enumerate(blocks, 1):
            name = status_rows[i-1]['candidate_name']
            slug = re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('._')[:80] or 'candidate'
            folder = f'candidates/{i:04d}_{slug}'
            rows = [dict(zip(FIELDS, [i, name, smiles[i-1], EXPERIMENTS[p[2]], p[0], p[1], p[3], p[4]])) for p in block]
            all_rows.extend(rows)
            metadata = dict(candidate_id=i, candidate_name=name, smiles=smiles[i-1], calculated_correlations=len(rows))
            for experiment in EXPERIMENTS.values():
                selected = [r for r in rows if r['experiment'] == experiment]
                archive.writestr(f'{folder}/{experiment}.csv', csv_bytes(selected, FIELDS))
                metadata[experiment + '_correlations'] = len(selected)
            selected_atoms = [r for r in atomic_rows if int(r['candidate_id']) == i]
            archive.writestr(f'{folder}/atomic_predictions.csv', csv_bytes(selected_atoms, atomic_fields))
            for nucleus in ('13C', '1H'):
                archive.writestr(f'{folder}/{nucleus}.csv', csv_bytes([r for r in selected_atoms if r['nucleus'] == nucleus], atomic_fields))
            metadata['atomic_status'] = status_rows[i-1]['status']
            index.append(metadata)
            archive.writestr(f'{folder}/structure.smi', smiles[i-1] + '\n')
        readable_fields = ['ID do candidato', 'Nome da substancia', 'SMILES', 'Experimento', 'Deslocamento 13C (ppm)', 'Deslocamento 1H (ppm)', 'Indice do atomo de carbono', 'Indice do atomo de hidrogenio']
        readable_rows = [dict(zip(readable_fields, [row[field] for field in FIELDS])) for row in all_rows]
        archive.writestr('organized_correlations.csv', csv_bytes(readable_rows, readable_fields))
        archive.writestr('resultprediction.csv', csv_bytes(atomic_rows, atomic_fields))
        archive.writestr('atomic_predictions.csv', csv_bytes(atomic_rows, atomic_fields))
        archive.writestr('atomic_prediction_status.csv', csv_bytes(status_rows, status_fields))
        archive.writestr('entries_without_prediction.csv', csv_bytes([r for r in status_rows if r['status'] != 'predicted'], status_fields))
        archive.writestr('calculated_correlations.csv', csv_bytes(all_rows, FIELDS))
        archive.writestr('candidate_index.csv', csv_bytes(index, ['candidate_id','candidate_name','smiles','calculated_correlations','HSQC_correlations','HMBC_correlations','HSQCTOCSY_correlations','atomic_status']))
        if source.exists(): archive.write(source, 'original/' + source.name)
        archive.write(atom_source, 'original/' + atom_source.name)
        archive.write(status_source, 'original/' + status_source.name)
        archive.write(project / 'nmrproc.properties', 'original/nmrproc.properties')
        structure_dir = project / 'result' / 'simulator_structures'
        if structure_dir.exists():
            for structure_file in sorted(structure_dir.glob('*')):
                if structure_file.is_file():
                    archive.write(structure_file, 'simulator_structures/' + structure_file.name)
        archive.writestr('README.txt', (
            'Complete NMRfilter atomic predictions\n\n'
            'resultprediction.csv and atomic_predictions.csv: one row per target atom, including missing predictions.\n'
            'Shifts come directly from the SAME PredictionTool.predict HOSE simulator as the 2D pipeline. No reconstruction from HSQC/HMBC.\n'
            'shift_ppm is the simulator mean (float array index 1); minimum/maximum and HOSE sphere count are retained.\n'
            'atom_index is one-based, matching legacy correlation numbering, after explicit hydrogen expansion; atom_id is its CDK identifier.\n'
            'These are simulator indices, not necessarily the atom order of an external SMILES parser.\n'
            'Missing shifts are blank, never zero or the simulator failure sentinel -1.\n'
            'All nonempty input lines have a candidate status; blank lines are ignored. Names use the optional names file, then the inline SMI name, then candidate_N.\n'
            'atomic_prediction_status.csv lists every entry; entries_without_prediction.csv includes partial predictions and errors.\n'
            'Per-candidate 13C.csv, 1H.csv and atomic_predictions.csv retain all target atoms without collapsing equivalent atoms.\n'
            'calculated_correlations.csv and organized_correlations.csv preserve the prior organized 2D results.\n'
            'original/ preserves unmodified simulator output and parameters.\n'
            f'Legacy correlations complete: {correlations_complete}. Partial legacy blocks are preserved only as raw output to prevent misidentification.\n'
        ))
    return bio.getvalue(), len(index), len(all_rows)
