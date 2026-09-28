# NMRfilter — Windows local startup (v7)

1. Extract the ZIP completely.
2. Double-click `START_NMRFILTER.bat`.
3. The launcher creates/updates the isolated `nmrfilter` environment (Python 3.11 + Java 17), validates dependencies, and starts **this package's `app.py`** on the dedicated address `http://127.0.0.1:8517`.
4. Keep the terminal open while using NMRfilter.

The launcher deliberately does not use port 8501, so another Streamlit application can remain open without being confused with NMRfilter. If startup fails, the terminal remains open and `nmrfilter_startup.log` records diagnostics.

To rebuild the environment, run `RESET_NMRFILTER_ENV.bat` and then `START_NMRFILTER.bat` again.
