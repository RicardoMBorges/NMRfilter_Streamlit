from __future__ import annotations
import csv, re
from pathlib import Path

VALID_TYPES = ('HMBC', 'HSQC', 'HSQCTOCSY')

def normalize_type(value: str) -> str:
    s = re.sub(r'[\s_\-]+', '', str(value).upper())
    if 'HSQC' in s and 'TOCSY' in s: return 'HSQCTOCSY'
    if 'HMBC' in s: return 'HMBC'
    if 'HSQC' in s: return 'HSQC'
    return ''

def _split(line: str):
    # Prefer explicit delimiters; whitespace fallback supports legacy text.
    for d in ('\t', ',', ';'):
        if d in line:
            return [x.strip() for x in next(csv.reader([line], delimiter=d, skipinitialspace=True))]
    return line.split()

def parse_measured_spectrum_text(text: str, default_type: str = ''):
    """Parse legacy sectioned or tab/comma/semicolon 2D peak lists.

    Accepted forms:
      HMBC\n120.1<TAB>7.2\nHSQC\n55.0<TAB>3.1
      13C,1H,type\n120.1,7.2,HMBC\n55.0,3.1,HSQC
    Returns ordered records (c, h, experiment). Untyped numeric rows are kept
    with experiment='' so the caller can reject scientifically ambiguous input.
    """
    default_type = normalize_type(default_type)
    records=[]; current_type=default_type; skipped=[]
    for lineno, raw in enumerate(text.splitlines(), 1):
        line=raw.strip().lstrip('\ufeff')
        if not line: continue
        marker=normalize_type(line)
        parts=_split(line)
        # A single-cell experiment marker changes the legacy section type.
        if marker and len(parts) == 1:
            current_type=marker; continue
        if len(parts) < 2:
            skipped.append((lineno, raw)); continue
        try:
            c=float(parts[0].replace(',', '.')) if len(parts)==2 and ';' in line else float(parts[0])
            h=float(parts[1].replace(',', '.')) if len(parts)==2 and ';' in line else float(parts[1])
        except ValueError:
            # Header or nonnumeric row.
            skipped.append((lineno, raw)); continue
        row_type = normalize_type(parts[2]) if len(parts) >= 3 else current_type
        records.append((c,h,row_type))
    return records, skipped

def parse_measured_spectrum_file(path, default_type=''):
    text=Path(path).read_text(encoding='utf-8-sig', errors='replace')
    return parse_measured_spectrum_text(text, default_type=default_type)

def to_legacy_tsv(records):
    """Canonical legacy representation, retaining experiment section markers."""
    out=[]; current=None
    for c,h,t in records:
        if t and t != current:
            out.append(t); current=t
        out.append(f'{c:.10g}\t{h:.10g}')
    return '\n'.join(out)+'\n'
