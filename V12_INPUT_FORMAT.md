# NMRfilter Streamlit v12 — measured spectrum input

The Streamlit wrapper now accepts TAB-, comma-, or semicolon-delimited measured peak lists.

For scientifically valid matching, every peak must have an experiment identity. Accepted forms:

## Legacy sectioned format
```
HMBC
120.1\t7.2
121.2\t7.3
HSQC
55.0\t3.1
```

## Three-column CSV
```
13C,1H,type
120.1,7.2,HMBC
121.2,7.3,HMBC
55.0,3.1,HSQC
```

Recognized types: `HMBC`, `HSQC`, `HSQCTOCSY` (common spacing/hyphen variants are normalized).

A two-column file with no section labels is rejected because the original NMRfilter similarity engine requires experiment identity to decide whether a measured peak can match an HMBC, HSQC, or HSQC-TOCSY prediction.
