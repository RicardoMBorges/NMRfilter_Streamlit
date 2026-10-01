"""Summarize HOSE matching specificity without assigning confidence probabilities."""
import pandas as pd

def summarize_hose(atoms, threshold=4):
    data = atoms.copy()
    spheres = pd.to_numeric(data['hose_spheres'], errors='coerce')
    predicted = data['prediction_status'].eq('predicted')
    data['low_hose'] = predicted & spheres.gt(0) & spheres.lt(threshold)
    data['unavailable'] = ~predicted | spheres.isna() | spheres.le(0)
    data['valid_spheres'] = spheres.where(predicted & spheres.gt(0))
    rows = []
    for (cid, name, smiles), group in data.groupby(['candidate_id','candidate_name','smiles'], sort=False, dropna=False):
        low = group[group['low_hose']]
        missing = group['unavailable'].sum()
        if low.empty and not missing:
            continue
        valid = group['valid_spheres'].dropna()
        rows.append(dict(candidate_id=cid, candidate_name=name, smiles=smiles,
                         minimum_hose_spheres=valid.min() if not valid.empty else None,
                         low_13C_atoms=int((low['nucleus']=='13C').sum()),
                         low_1H_atoms=int((low['nucleus']=='1H').sum()),
                         atoms_without_hose=int(missing), total_target_atoms=len(group)))
    fields = ['candidate_id','candidate_name','smiles','minimum_hose_spheres','low_13C_atoms','low_1H_atoms','atoms_without_hose','total_target_atoms']
    return pd.DataFrame(rows, columns=fields), data[data['low_hose'] | data['unavailable']].drop(columns=['low_hose','unavailable','valid_spheres'])
