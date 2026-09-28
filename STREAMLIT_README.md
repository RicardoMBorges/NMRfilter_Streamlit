# NMRfilter Streamlit wrapper

This package rebuilds the UI around the original NMRfilter v1.5 processing engine.

## Streamlit Community Cloud
- Main file: `app.py`
- `requirements.txt` contains Python dependencies.
- `packages.txt` installs a headless Java runtime.
- `runtime.txt` requests Python 3.11.

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Java must be installed and available as `java`.

## Pipeline
1. Prepare an isolated project folder.
2. Run original `nmrfilter.py`.
3. Run `uk.ac.dmu.simulate.Convert` from `lib/simulate.jar`.
4. Run `uk.ac.dmu.simulate.Simulate`.
5. Run original `nmrfilter2.py` for clustering, Louvain communities and ranking.

Bruker backgrounds are disabled by default and do not affect ranking. Respredict/deep learning is disabled in this first Streamlit wrapper; the original HOSE-code prediction path is used.
