# NMRfilter Streamlit

## Windows local — recommended

1. Extract the ZIP completely.
2. Double-click `START_NMRFILTER.bat`.
3. On first run it creates an isolated Conda environment named `nmrfilter` with Python 3.11 and Java 17, installs `requirements.txt`, validates `igraph` + `leidenalg`, then launches Streamlit with that interpreter.
4. Later runs reuse the same environment and verify/update dependencies before launch.

Do **not** start this project with the base Miniconda Python (`...\miniconda3\python.exe`) or with a globally installed `streamlit`; those may be Python 3.13 and do not define the NMRfilter runtime.

If the environment becomes inconsistent, run `RESET_NMRFILTER_ENV.bat` and then `START_NMRFILTER.bat` again.

## Streamlit Community Cloud

Use `app.py` as the main file. `runtime.txt`, `requirements.txt`, and `packages.txt` define the cloud runtime.
