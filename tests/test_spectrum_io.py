import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectrum_io import parse_measured_spectrum_text, to_legacy_tsv

def test_legacy_sections():
    r,_=parse_measured_spectrum_text('HMBC\n120.1\t7.2\nHSQC\n55.0\t3.1\n')
    assert r == [(120.1,7.2,'HMBC'),(55.0,3.1,'HSQC')]

def test_csv_type_column():
    r,_=parse_measured_spectrum_text('13C,1H,type\n120.1,7.2,HMBC\n55.0,3.1,HSQC\n')
    assert r == [(120.1,7.2,'HMBC'),(55.0,3.1,'HSQC')]

def test_untyped_detectable():
    r,_=parse_measured_spectrum_text('120.1,7.2\n55.0,3.1\n')
    assert len(r)==2 and all(not x[2] for x in r)

def test_canonical_roundtrip():
    src=[(120.1,7.2,'HMBC'),(121.2,7.3,'HMBC'),(55.0,3.1,'HSQC')]
    r,_=parse_measured_spectrum_text(to_legacy_tsv(src))
    assert r == src

if __name__=='__main__':
    for f in (test_legacy_sections,test_csv_type_column,test_untyped_detectable,test_canonical_roundtrip): f()
    print('SPECTRUM PARSER TESTS: PASS')


def test_two_column_default_hmbc():
    from spectrum_io import parse_measured_spectrum_text
    r,_=parse_measured_spectrum_text("178.55\t8.23\n163.44\t7.93\n", default_type="HMBC")
    assert r == [(178.55,8.23,"HMBC"),(163.44,7.93,"HMBC")]

def test_explicit_marker_overrides_default():
    from spectrum_io import parse_measured_spectrum_text
    r,_=parse_measured_spectrum_text("HSQC\n55.0\t3.1\n", default_type="HMBC")
    assert r == [(55.0,3.1,"HSQC")]
