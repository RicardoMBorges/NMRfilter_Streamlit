# app.py for NMRfilter

from __future__ import annotations

import configparser
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid
import zipfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE
# --------------------------------------------------
st.set_page_config(page_title="NMRfilter Online", layout="wide")

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR / "nmrfilter-master"
BACKEND_ZIP = APP_DIR / "nmrfilter-master.zip"
WORKSPACE_DIR = APP_DIR / "_nmrfilter_web_projects"
WORKSPACE_DIR.mkdir(exist_ok=True)

SOLVENT_OPTIONS = [
    "Methanol-D4 (CD3OD)",
    "Chloroform-D1 (CDCl3)",
    "Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)",
    "Unreported",
]

CORE_BACKEND_ITEMS = [
    "nmrfilter.py",
    "nmrfilter2.py",
    "nmrutil.py",
    "clustering.py",
    "clusterlouvain.py",
    "similarity.py",
    "plotutil.py",
    "nmrproc.properties",
    "simulate.jar",
    "DumpParser2-1.4.jar",
    "glossary.csv",
    "LICENSE",
]

OPTIONAL_BACKEND_DIRS = ["lib", "respredict"]


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or f"project_{uuid.uuid4().hex[:8]}"


@st.cache_resource(show_spinner=False)
def ensure_backend() -> Path:
    if BACKEND_DIR.exists():
        return BACKEND_DIR

    if not BACKEND_ZIP.exists():
        raise FileNotFoundError(
            "Could not find 'nmrfilter-master' folder or 'nmrfilter-master.zip' next to app.py."
        )

    with zipfile.ZipFile(BACKEND_ZIP, "r") as zf:
        zf.extractall(APP_DIR)

    if not BACKEND_DIR.exists():
        raise FileNotFoundError("The backend ZIP was extracted, but 'nmrfilter-master' was not created.")

    return BACKEND_DIR


@st.cache_data(show_spinner=False)
def read_default_properties() -> dict:
    backend = ensure_backend()
    cp = configparser.ConfigParser()
    cp.read(backend / "nmrproc.properties")
    if "onesectiononly" not in cp:
        return {}
    return dict(cp["onesectiononly"])


@st.cache_data(show_spinner=False)
def load_readme() -> str:
    backend = ensure_backend()
    readme = backend / "README.md"
    return readme.read_text(encoding="utf-8", errors="ignore") if readme.exists() else ""


@st.cache_data(show_spinner=False)
def java_available() -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["java", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return proc.returncode == 0, proc.stdout
    except Exception as exc:
        return False, str(exc)


@st.cache_data(show_spinner=False)
def python_runtime() -> str:
    return sys.executable


@st.cache_data(show_spinner=False)
def packages_hint() -> str:
    return textwrap.dedent(
        """
        For Streamlit Community Cloud or another Linux deployment, include a file named packages.txt with:

        default-jre
        default-jdk
        """
    ).strip()



def prepare_runtime_tree(run_root: Path) -> Path:
    backend = ensure_backend()
    runtime_dir = run_root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for item in CORE_BACKEND_ITEMS:
        src = backend / item
        dst = runtime_dir / item
        if not src.exists():
            continue
        if src.is_file():
            shutil.copy2(src, dst)

    for dirname in OPTIONAL_BACKEND_DIRS:
        src = backend / dirname
        dst = runtime_dir / dirname
        if not src.exists():
            continue
        try:
            os.symlink(src, dst, target_is_directory=True)
        except Exception:
            if src.is_dir():
                shutil.copytree(src, dst)

    return runtime_dir



def write_project_inputs(
    project_dir: Path,
    smiles_text: str,
    spectrum_bytes: bytes,
    spectrum_name: str,
    names_text: str | None,
    solvent: str,
    tolerancec: float,
    toleranceh: float,
    usehmbc: bool,
    usehsqctocsy: bool,
    debug: bool,
    labelsimulated: bool,
    dotwobonds: bool,
    usedeeplearning: bool,
    workspace_dir: Path,
) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)

    (project_dir / "testall.smi").write_text(smiles_text.strip() + "\n", encoding="utf-8")
    (project_dir / "realspectrum.csv").write_bytes(spectrum_bytes)

    if names_text and names_text.strip():
        (project_dir / "testallnames.txt").write_text(names_text.strip() + "\n", encoding="utf-8")

    cp = configparser.ConfigParser()
    cp["onesectiononly"] = {
        "datadir": workspace_dir.resolve().as_posix(),
        "msmsinput": "testall.smi",
        "predictionoutput": "resultprediction.csv",
        "result": "result.txt",
        "solvent": solvent,
        "tolerancec": str(tolerancec),
        "toleranceh": str(toleranceh),
        "spectruminput": "realspectrum.csv",
        "clusteringoutput": "cluster.txt",
        "rberresolution": "0.2",
        "louvainoutput": "clusterslouvain.txt",
        "usehsqctocsy": str(usehsqctocsy).lower(),
        "usehmbc": str(usehmbc).lower(),
        "dotwobonds": str(dotwobonds).lower(),
        "usedeeplearning": str(usedeeplearning).lower(),
        "debug": str(debug).lower(),
        "labelsimulated": str(labelsimulated).lower(),
        "hmbcbruker": "NaN",
        "hsqcbruker": "NaN",
        "hsqctocsybruker": "NaN",
    }

    with open(project_dir / "nmrproc.properties", "w", encoding="utf-8") as fh:
        cp.write(fh)



def save_uploaded_project_zip(project_zip_bytes: bytes, run_root: Path) -> Tuple[str, Path]:
    incoming_dir = run_root / "uploaded_project"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    zip_path = incoming_dir / "project.zip"
    zip_path.write_bytes(project_zip_bytes)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(incoming_dir)

    children = [p for p in incoming_dir.iterdir() if p.name != "project.zip"]
    if len(children) == 1 and children[0].is_dir():
        source_dir = children[0]
    else:
        source_dir = incoming_dir

    project_name = safe_name(source_dir.name)
    final_project_dir = WORKSPACE_DIR / project_name
    if final_project_dir.exists():
        shutil.rmtree(final_project_dir)
    shutil.copytree(source_dir, final_project_dir)
    return project_name, final_project_dir



def read_text_file(uploaded_file) -> str:
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")



def run_nmrfilter_pipeline(runtime_dir: Path, project_name: str, simulate_only: bool = False) -> Tuple[bool, str]:
    logs: List[str] = []

    def run_step(cmd: List[str], title: str) -> None:
        logs.append(f"\n### {title}\n$ {' '.join(cmd)}\n")
        proc = subprocess.run(
            cmd,
            cwd=runtime_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        logs.append(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"Step failed: {title}\n\n{proc.stdout}")

    py = sys.executable
    java_cp = f"simulate.jar{os.pathsep}lib/*"

    run_step([py, "nmrfilter.py", project_name], "Validate project and initialize folders")

    convert_proc = subprocess.run(
        ["java", "-cp", java_cp, "uk.ac.dmu.simulate.Convert", project_name],
        cwd=runtime_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    logs.append(
        f"\n### Generate candidate prediction inputs\n$ java -cp {java_cp} uk.ac.dmu.simulate.Convert {project_name}\n"
    )
    logs.append(convert_proc.stdout)
    if convert_proc.returncode != 0:
        raise RuntimeError(convert_proc.stdout)

    run_step(
        ["java", "-cp", java_cp, "uk.ac.dmu.simulate.Simulate", project_name],
        "Simulate spectra",
    )

    if not simulate_only:
        run_step([py, "nmrfilter2.py", project_name], "Cluster peaks and rank candidates")

    return True, "\n".join(logs)



def collect_output_files(project_dir: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in [
        "result/*.txt",
        "result/*.csv",
        "result/*.json",
        "result/*.sdf",
        "plots/*.png",
        "sim_plots/*.png",
    ]:
        files.extend(sorted(project_dir.glob(pattern)))
    return files



def build_output_zip(project_dir: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in project_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(project_dir))
    return zip_path



def render_results(project_dir: Path) -> None:
    result_txt = project_dir / "result" / "result.txt"
    result_csv = project_dir / "result" / "resultprediction.csv"
    plots_dir = project_dir / "plots"
    sim_plots_dir = project_dir / "sim_plots"

    c1, c2 = st.columns([1.2, 0.8])

    with c1:
        st.subheader("Ranking")
        if result_txt.exists():
            st.code(result_txt.read_text(encoding="utf-8", errors="ignore"), language="text")
        else:
            st.info("No result.txt was generated.")

        if result_csv.exists():
            st.subheader("Predicted shifts table")
            try:
                df = pd.read_csv(result_csv)
                st.dataframe(df, use_container_width=True)
            except Exception:
                st.download_button(
                    "Download resultprediction.csv",
                    data=result_csv.read_bytes(),
                    file_name="resultprediction.csv",
                    mime="text/csv",
                )

    with c2:
        st.subheader("Plots")
        plot_files = []
        if plots_dir.exists():
            plot_files.extend(sorted(plots_dir.glob("*.png")))
        if sim_plots_dir.exists():
            plot_files.extend(sorted(sim_plots_dir.glob("*.png")))

        if not plot_files:
            st.info("No PNG plots were found. Add testallnames.txt if you want labeled plot outputs.")
        else:
            for img in plot_files:
                st.image(str(img), caption=img.name, use_container_width=True)

import pathlib

def patch_nmrfilter(runtime_dir: Path):
    for file in runtime_dir.rglob("*.py"):
        text = file.read_text(encoding="utf-8", errors="ignore")

        new_text = text

        # Fix old ConfigParser constructor usage
        new_text = new_text.replace(
            "configparser.SafeConfigParser()",
            "configparser.ConfigParser()"
        )
        new_text = new_text.replace(
            "SafeConfigParser()",
            "ConfigParser()"
        )

        # Fix removed readfp() calls
        new_text = new_text.replace(
            "cp.readfp(open('nmrproc.properties'))",
            "with open('nmrproc.properties', 'r', encoding='utf-8', errors='ignore') as fh:\n\t\tcp.read_file(fh)"
        )

        new_text = new_text.replace(
            "cp2.readfp(open(datapath+os.sep+project+os.sep+'nmrproc.properties'))",
            "with open(datapath+os.sep+project+os.sep+'nmrproc.properties', 'r', encoding='utf-8', errors='ignore') as fh:\n\t\t\tcp2.read_file(fh)"
        )

        if new_text != text:
            file.write_text(new_text, encoding="utf-8")
            
def force_replace_nmrutil(runtime_dir: Path):
    nmrutil_path = runtime_dir / "nmrutil.py"
    nmrutil_code = """import configparser
import os


def readprops(project=""):
\tresult = {}
\tcp = configparser.ConfigParser()

\twith open("nmrproc.properties", "r", encoding="utf-8", errors="ignore") as fh:
\t\tcp.read_file(fh)

\tfor (each_key, each_val) in cp.items("onesectiononly"):
\t\tresult[each_key] = each_val

\tdatapath = cp.get("onesectiononly", "datadir")

\tproject_props = datapath + os.sep + project + os.sep + "nmrproc.properties"
\tif project != "" and os.path.exists(project_props):
\t\tcp2 = configparser.ConfigParser()
\t\twith open(project_props, "r", encoding="utf-8", errors="ignore") as fh:
\t\t\tcp2.read_file(fh)

\t\tfor (each_key, each_val) in cp2.items("onesectiononly"):
\t\t\tresult[each_key] = each_val

\treturn result


def checkprojectdir(datapath, project, cp):
\tif not os.path.exists(datapath + os.sep + project):
\t\tprint("There is no directory " + datapath + os.sep + project + " - please check!")

\tif os.path.exists(datapath + os.sep + project + os.sep + "result"):
\t\tpredictionoutputfile = datapath + os.sep + project + os.sep + 'result' + os.sep + cp.get('predictionoutput')
\t\tif os.path.exists(predictionoutputfile):
\t\t\tos.remove(predictionoutputfile)
\t\tclusteringoutputfile = datapath + os.sep + project + os.sep + 'result' + os.sep + cp.get('clusteringoutput')
\t\tif os.path.exists(clusteringoutputfile):
\t\t\tos.remove(clusteringoutputfile)
\t\tlouvainoutputfile = datapath + os.sep + project + os.sep + 'result' + os.sep + cp.get('louvainoutput')
\t\tif os.path.exists(louvainoutputfile):
\t\t\tos.remove(louvainoutputfile)
\t\tpredictionoutputfile = datapath + os.sep + project + os.sep + 'result' + os.sep + cp.get('predictionoutput') + 'hsqc'
\t\tif os.path.exists(predictionoutputfile):
\t\t\tos.remove(predictionoutputfile)
\t\tpredictionoutputfile = datapath + os.sep + project + os.sep + 'result' + os.sep + cp.get('predictionoutput') + 'hmbc'
\t\tif os.path.exists(predictionoutputfile):
\t\t\tos.remove(predictionoutputfile)
\t\tpredictionoutputfile = datapath + os.sep + project + os.sep + 'result' + os.sep + cp.get('predictionoutput') + 'hsqctocsy'
\t\tif os.path.exists(predictionoutputfile):
\t\t\tos.remove(predictionoutputfile)
\telse:
\t\tos.mkdir(datapath + os.sep + project + os.sep + "result")

\tif not os.path.exists(datapath + os.sep + project + os.sep + "result" + os.sep + "smart"):
\t\tos.mkdir(datapath + os.sep + project + os.sep + "result" + os.sep + "smart")

\tif os.path.exists(datapath + os.sep + project + os.sep + "plots"):
\t\tfor f in os.listdir(datapath + os.sep + project + os.sep + "plots"):
\t\t\tif f.endswith(".png"):
\t\t\t\tos.remove(os.path.join(datapath + os.sep + project + os.sep + "plots", f))
\telse:
\t\tos.mkdir(datapath + os.sep + project + os.sep + "plots")
"""
    nmrutil_path.write_text(nmrutil_code, encoding="utf-8")

def write_runtime_properties(
    runtime_dir: Path,
    workspace_dir: Path,
    solvent: str,
    tolerancec: float,
    toleranceh: float,
    usehmbc: bool,
    usehsqctocsy: bool,
    debug: bool,
    labelsimulated: bool,
    dotwobonds: bool,
    usedeeplearning: bool,
) -> None:
    cp = configparser.ConfigParser()
    cp["onesectiononly"] = {
        "datadir": workspace_dir.resolve().as_posix(),
        "msmsinput": "testall.smi",
        "predictionoutput": "resultprediction.csv",
        "result": "result.txt",
        "solvent": solvent,
        "tolerancec": str(tolerancec),
        "toleranceh": str(toleranceh),
        "spectruminput": "realspectrum.csv",
        "clusteringoutput": "cluster.txt",
        "rberresolution": "0.2",
        "louvainoutput": "clusterslouvain.txt",
        "usehsqctocsy": str(usehsqctocsy).lower(),
        "usehmbc": str(usehmbc).lower(),
        "dotwobonds": str(dotwobonds).lower(),
        "usedeeplearning": str(usedeeplearning).lower(),
        "debug": str(debug).lower(),
        "labelsimulated": str(labelsimulated).lower(),
        "hmbcbruker": "NaN",
        "hsqcbruker": "NaN",
        "hsqctocsybruker": "NaN",
    }

    with open(runtime_dir / "nmrproc.properties", "w", encoding="utf-8") as fh:
        cp.write(fh)

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("NMRfilter Online")
st.caption("Streamlit interface for running NMRfilter projects in a browser.")

with st.expander("What this app expects", expanded=False):
    st.markdown(
        """
        The NMRfilter backend needs a project folder containing at least:

        - `testall.smi`: one candidate SMILES per line
        - `realspectrum.csv`: experimental 2D NMR peaks as tab-separated `13C` and `1H` coordinates
        - `testallnames.txt` (recommended): one compound name per line, in the same order as the SMILES list
        """
    )
    readme = load_readme()
    if readme:
        st.text_area("Backend README excerpt", readme[:8000], height=220)

backend_ok = False
backend_error = None
try:
    ensure_backend()
    backend_ok = True
except Exception as exc:
    backend_error = str(exc)

java_ok, java_msg = java_available()

status_col1, status_col2, status_col3 = st.columns(3)
status_col1.metric("Backend", "Found" if backend_ok else "Missing")
status_col2.metric("Java", "OK" if java_ok else "Missing")
status_col3.metric("Python", Path(python_runtime()).name)

if backend_error:
    st.error(backend_error)

if not java_ok:
    st.warning("Java is not available in this environment. The app will not run until Java is installed.")
    st.code(packages_hint(), language="text")
    with st.expander("Java detection details"):
        st.text(java_msg)

with st.sidebar:
    st.header("Input mode")
    input_mode = st.radio(
        "Choose how to provide the project",
        ["Upload individual files", "Upload a whole project ZIP"],
        index=0,
    )

    st.header("NMRfilter settings")
    defaults = read_default_properties() if backend_ok else {}
    solvent = st.selectbox(
        "Solvent",
        SOLVENT_OPTIONS,
        index=SOLVENT_OPTIONS.index(defaults.get("solvent", SOLVENT_OPTIONS[0]))
        if defaults.get("solvent") in SOLVENT_OPTIONS
        else 0,
    )
    tolerancec = st.number_input("13C tolerance", min_value=0.0, value=float(defaults.get("tolerancec", 0.2)), step=0.01)
    toleranceh = st.number_input("1H tolerance", min_value=0.0, value=float(defaults.get("toleranceh", 0.02)), step=0.001, format="%.3f")
    usehmbc = st.checkbox("Use HMBC", value=str(defaults.get("usehmbc", "true")).lower() == "true")
    usehsqctocsy = st.checkbox("Use HSQCTOCSY", value=str(defaults.get("usehsqctocsy", "false")).lower() == "true")
    debug = st.checkbox("Debug mode", value=str(defaults.get("debug", "false")).lower() == "true")
    labelsimulated = st.checkbox("Label simulated peaks", value=str(defaults.get("labelsimulated", "true")).lower() == "true")
    dotwobonds = st.checkbox("Use 2-sphere HOSE instead of 3", value=str(defaults.get("dotwobonds", "false")).lower() == "true")
    usedeeplearning = st.checkbox(
        "Use respredict deep learning backend",
        value=str(defaults.get("usedeeplearning", "false")).lower() == "true",
        help="This needs the respredict dependencies and models present in the backend folder.",
    )

main_left, main_right = st.columns([1.1, 0.9])

project_name_input = main_left.text_input("Project name", value="demo_project")
project_name = safe_name(project_name_input)

simulate_only = main_left.checkbox(
    "Simulate spectra only",
    value=False,
    help="Runs the simulation stage but skips the final ranking step.",
)

if input_mode == "Upload individual files":
    smiles_file = main_left.file_uploader("Candidate SMILES file (.smi or .txt)", type=["smi", "txt"])
    spectrum_file = main_left.file_uploader("Measured spectrum file (.csv or .tsv)", type=["csv", "tsv", "txt"])
    names_file = main_left.file_uploader("Optional names file (.txt)", type=["txt"])
    project_zip = None
else:
    project_zip = main_left.file_uploader("Project ZIP", type=["zip"])
    smiles_file = None
    spectrum_file = None
    names_file = None

main_right.markdown("### Deployment notes")
main_right.markdown(
    """
    Put this `app.py` next to either:

    - a folder named `nmrfilter-master`, or
    - the file `nmrfilter-master.zip`

    For cloud deployment, Java is usually the extra missing dependency.
    """
)
main_right.code(packages_hint(), language="text")

run_clicked = st.button("Run NMRfilter", type="primary", use_container_width=True)

if run_clicked:
    if not backend_ok:
        st.stop()
    if not java_ok:
        st.stop()

    try:
        run_id = f"{project_name}_{uuid.uuid4().hex[:8]}"
        run_root = Path(tempfile.mkdtemp(prefix=run_id + "_", dir=WORKSPACE_DIR))
        runtime_dir = prepare_runtime_tree(run_root)
        patch_nmrfilter(runtime_dir)
        force_replace_nmrutil(runtime_dir)

        write_runtime_properties(
            runtime_dir=runtime_dir,
            workspace_dir=WORKSPACE_DIR,
            solvent=solvent,
            tolerancec=float(tolerancec),
            toleranceh=float(toleranceh),
            usehmbc=usehmbc,
            usehsqctocsy=usehsqctocsy,
            debug=debug,
            labelsimulated=labelsimulated,
            dotwobonds=dotwobonds,
            usedeeplearning=usedeeplearning,
        )

        patched_file = runtime_dir / "nmrutil.py"
        if patched_file.exists():
            st.code(patched_file.read_text(encoding="utf-8", errors="ignore"), language="python")

        if project_zip is not None:
            uploaded_project_name, uploaded_project_dir = save_uploaded_project_zip(project_zip.getvalue(), run_root)
            project_name = uploaded_project_name
            project_dir = uploaded_project_dir
        else:
            if smiles_file is None or spectrum_file is None:
                st.error("Please upload at least the SMILES file and the measured spectrum file.")
                st.stop()

            project_dir = WORKSPACE_DIR / project_name
            if project_dir.exists():
                shutil.rmtree(project_dir)

            write_project_inputs(
                project_dir=project_dir,
                smiles_text=read_text_file(smiles_file),
                spectrum_bytes=spectrum_file.getvalue(),
                spectrum_name=spectrum_file.name,
                names_text=read_text_file(names_file) if names_file is not None else None,
                solvent=solvent,
                tolerancec=float(tolerancec),
                toleranceh=float(toleranceh),
                usehmbc=usehmbc,
                usehsqctocsy=usehsqctocsy,
                debug=debug,
                labelsimulated=labelsimulated,
                dotwobonds=dotwobonds,
                usedeeplearning=usedeeplearning,
                workspace_dir=WORKSPACE_DIR,
            )

        with st.spinner("Running NMRfilter. This may take a while depending on the number of candidates."):
            ok, logs = run_nmrfilter_pipeline(runtime_dir, project_name, simulate_only=simulate_only)

        if ok:
            st.success("Run completed.")
            render_results(project_dir)

            log_path = run_root / "run.log"
            log_path.write_text(logs, encoding="utf-8")
            outputs_zip = build_output_zip(project_dir, run_root / f"{project_name}_outputs.zip")

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download outputs ZIP",
                    data=outputs_zip.read_bytes(),
                    file_name=outputs_zip.name,
                    mime="application/zip",
                    use_container_width=True,
                )
            with d2:
                st.download_button(
                    "Download run log",
                    data=log_path.read_bytes(),
                    file_name=log_path.name,
                    mime="text/plain",
                    use_container_width=True,
                )

            with st.expander("Execution log"):
                st.code(logs, language="text")

    except Exception as exc:
        st.error(f"NMRfilter failed: {exc}")
