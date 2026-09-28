import csv
import os
import time


def Two_Column_List_c(file):
    from spectrum_io import parse_measured_spectrum_file
    records, _ = parse_measured_spectrum_file(file)
    return [[i, c, h] for i, (c, h, _type) in enumerate(records)]


def setofy(peaks):
    return {peak[2] for peak in peaks}


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _legacy_anchor_bins(peaks, axis, tolerance):
    """Reproduce the legacy first-peak anchored binning exactly.

    The original NMRfilter does not perform pairwise/transitive tolerance
    clustering here. Each new peak is assigned to the first existing bin whose
    *first peak* is within the strict tolerance; otherwise it starts a new bin.
    Peak order is therefore intentionally preserved.
    """
    bins = []
    for peak in peaks:
        found = False
        for group in bins:
            anchor = group[0][axis]
            if peak[axis] > anchor - tolerance and peak[axis] < anchor + tolerance:
                group.append(peak)
                found = True
                break
        if not found:
            bins.append([peak])
    return bins


def _merge_xbins_via_ybins(xbins, ybins):
    """Compute the intended closure of the legacy merge loop, deterministically.

    A proton bin bridges every carbon bin containing one of its member peaks.
    The old ``while found`` repeatedly merged pairs of carbon bins and could
    fail to converge. Union-find is applied to *legacy carbon bins*, not raw
    peaks, preserving the original two-stage semantics without the loop.
    """
    peak_to_xbin = {}
    for xi, group in enumerate(xbins):
        for peak in group:
            peak_to_xbin[peak[0]] = xi

    uf = _UnionFind(len(xbins))
    bridge_count = 0
    for ygroup in ybins:
        touched = []
        seen = set()
        for peak in ygroup:
            xi = peak_to_xbin[peak[0]]
            if xi not in seen:
                seen.add(xi)
                touched.append(xi)
        if len(touched) > 1:
            root = touched[0]
            for xi in touched[1:]:
                if uf.union(root, xi):
                    bridge_count += 1

    merged = {}
    for xi, group in enumerate(xbins):
        merged.setdefault(uf.find(xi), []).extend(group)
    clusters = list(merged.values())
    clusters.sort(key=lambda c: min(p[0] for p in c))
    return clusters, bridge_count


def cluster2dspectrum(cp, project):
    """Cluster measured 2D peaks using the intended legacy NMRfilter semantics.

    v9 incorrectly applied union-find directly to raw peak pairs satisfying
    dC OR dH tolerance, making tolerance transitive at peak level and producing
    giant components. This implementation restores the original algorithm's
    two-stage meaning:
      1) first-peak anchored carbon bins;
      2) first-peak anchored proton bins;
      3) merge carbon bins bridged by proton bins.

    Only the non-convergent pairwise ``while found`` is replaced, by a finite
    union-find closure over the already-created carbon bins.
    """
    t0 = time.perf_counter()
    datapath = cp.get('datadir')
    c_limit = float(cp.get('tolerancec'))
    h_limit = float(cp.get('toleranceh'))
    spectrum_path = os.path.join(datapath, project, cp.get('spectruminput'))
    output_path = os.path.join(datapath, project, 'result', cp.get('clusteringoutput'))

    print(f"[cluster] Reading measured peaks: {spectrum_path}", flush=True)
    peaks = Two_Column_List_c(spectrum_path)
    print(f"[cluster] Loaded {len(peaks)} peaks; tolerances dC={c_limit}, dH={h_limit}", flush=True)
    if not peaks:
        raise ValueError("Measured spectrum contains 0 readable peaks. Expected numeric 13C/1H coordinates separated by TAB, comma, or semicolon.")

    b0 = time.perf_counter()
    xbins = _legacy_anchor_bins(peaks, 1, c_limit)
    ybins = _legacy_anchor_bins(peaks, 2, h_limit)
    print(f"[cluster] Legacy anchor bins: {len(xbins)} carbon, {len(ybins)} proton ({time.perf_counter()-b0:.3f} s)", flush=True)

    clusters, bridge_count = _merge_xbins_via_ybins(xbins, ybins)
    sizes = sorted((len(c) for c in clusters), reverse=True)
    print(f"[cluster] Carbon-bin bridges: {bridge_count}; final clusters: {len(clusters)}; largest sizes={sizes[:10]}", flush=True)

    edge_count = 0
    with open(output_path, 'w', newline='') as f:
        # Preserve the exact legacy edge predicate (inclusive bounds here).
        for cluster in clusters:
            for peak1 in cluster:
                for peak2 in cluster:
                    if peak2[0] >= peak1[0] and (
                        (peak1[1] >= peak2[1] - c_limit and peak1[1] <= peak2[1] + c_limit)
                        or
                        (peak1[2] >= peak2[2] - h_limit and peak1[2] <= peak2[2] + h_limit)
                    ):
                        f.write(f"{peak1[0]} {peak2[0]}\n")
                        edge_count += 1

    print(f"[cluster] Wrote {edge_count:,} edges to {output_path}", flush=True)
    print(f"[cluster] Finished in {time.perf_counter()-t0:.2f} s", flush=True)
    return clusters
