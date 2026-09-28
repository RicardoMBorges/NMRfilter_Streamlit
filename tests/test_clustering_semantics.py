"""Known-positive regression tests for NMRfilter legacy clustering semantics."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clustering import _legacy_anchor_bins, _merge_xbins_via_ybins, cluster2dspectrum


def ids(groups):
    return [sorted(p[0] for p in g) for g in groups]


def test_anchor_binning_is_not_raw_pairwise_transitive():
    # C0~C1 and C1~C2 pairwise, but C2 is outside the tolerance of the first
    # bin anchor C0. Legacy NMRfilter therefore makes TWO carbon bins.
    peaks = [[0, 10.00, 1.00], [1, 10.15, 2.00], [2, 10.30, 3.00]]
    xbins = _legacy_anchor_bins(peaks, 1, 0.20)
    assert ids(xbins) == [[0, 1], [2]], ids(xbins)


def test_proton_bin_bridges_legacy_carbon_bins():
    # Peaks 0/1 form one carbon bin; 2/3 another. Peaks 1 and 2 share a proton
    # bin, so the intended legacy merge joins the two carbon bins.
    peaks = [
        [0, 10.00, 1.000],
        [1, 10.10, 2.000],
        [2, 20.00, 2.010],
        [3, 20.10, 3.000],
        [4, 40.00, 8.000],
    ]
    xbins = _legacy_anchor_bins(peaks, 1, 0.20)
    ybins = _legacy_anchor_bins(peaks, 2, 0.02)
    merged, bridges = _merge_xbins_via_ybins(xbins, ybins)
    assert ids(merged) == [[0, 1, 2, 3], [4]], ids(merged)
    assert bridges == 1


def test_end_to_end_known_positive_edges():
    peaks = [
        (10.00, 1.000),
        (10.10, 2.000),
        (20.00, 2.010),
        (20.10, 3.000),
        (40.00, 8.000),
    ]
    with tempfile.TemporaryDirectory() as td:
        project = 'known_positive'
        p = Path(td) / project
        (p / 'result').mkdir(parents=True)
        with open(p / 'realspectrum.csv', 'w') as f:
            for c, h in peaks:
                f.write(f'{c}\t{h}\n')
        cp = {
            'datadir': td,
            'tolerancec': '0.2',
            'toleranceh': '0.02',
            'spectruminput': 'realspectrum.csv',
            'clusteringoutput': 'cluster.txt',
        }
        clusters = cluster2dspectrum(cp, project)
        assert ids(clusters) == [[0, 1, 2, 3], [4]], ids(clusters)
        edges = set((p / 'result' / 'cluster.txt').read_text().splitlines())
        # Self edges preserve every peak; these positive links are expected.
        assert {'0 0', '0 1', '1 1', '1 2', '2 2', '2 3', '3 3', '4 4'} <= edges


if __name__ == '__main__':
    test_anchor_binning_is_not_raw_pairwise_transitive()
    test_proton_bin_bridges_legacy_carbon_bins()
    test_end_to_end_known_positive_edges()
    print('KNOWN-POSITIVE CLUSTERING TESTS: PASS')
