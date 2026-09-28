# NMRfilter v11 — restored legacy clustering semantics

## What was wrong in v9/v10
v9 replaced the non-convergent legacy merge loop with union-find directly over raw peaks connected by `dC OR dH`. That made tolerance transitive at the individual-peak level and could collapse most measured peaks into a giant component.

## What v11 restores
The original two-stage semantics are retained:
1. Build carbon bins in input order. Membership is tested against the **first peak (anchor)** of each existing bin using the original strict carbon tolerance.
2. Build proton bins the same way using the original strict proton tolerance.
3. Merge carbon bins when a proton bin contains peaks belonging to more than one carbon bin.
4. Write graph edges using the original inclusive `dC OR dH` edge predicate inside the resulting merged carbon clusters.

Only step 3's potentially non-convergent `while found` implementation is replaced. Union-find now operates on the already-created **carbon bins**, not on raw peaks.

## Known-positive regression validation
`tests/test_clustering_semantics.py` contains three checks:
- a chained carbon-tolerance example proving that anchor bins are not raw pairwise-transitive;
- a positive proton bridge that must merge two carbon bins;
- an end-to-end five-peak example checking the expected clusters and graph edges.

Run on Windows with `RUN_CLUSTER_VALIDATION.bat`.
Expected final line: `KNOWN-POSITIVE CLUSTERING TESTS: PASS`.
