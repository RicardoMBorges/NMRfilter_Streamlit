from __future__ import annotations
import configparser, io, os, re, shutil, subprocess, sys, tempfile, time, uuid, zipfile
from pathlib import Path
import streamlit as st
import pandas as pd
from spectrum_io import parse_measured_spectrum_text, to_legacy_tsv
from prediction_export import build_prediction_zip

ROOT = Path(__file__).resolve().parent
PROJECTS = ROOT / 'streamlit_projects'

def java_property_path(path: Path) -> str:
    """Return an absolute path safe for java.util.Properties on Windows/Linux.

    Backslashes in .properties are escape characters; Path.as_posix() avoids
    Java turning C:\\Users\\... into C:Users....
    """
    return path.resolve().as_posix()
PROJECTS.mkdir(exist_ok=True)

st.set_page_config(page_title='NMRfilter', page_icon='🧪', layout='wide')

SOLVENTS=['Methanol-D4 (CD3OD)','Chloroform-D1 (CDCl3)','Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)','Unreported']

def safe_name(s):
    s=re.sub(r'[^A-Za-z0-9_.-]+','_',s.strip())
    return s.strip('._') or 'project'

def write_props(project_dir, opts):
    cp=configparser.ConfigParser()
    cp['onesectiononly']={
      'datadir': java_property_path(PROJECTS), 'msmsinput':'testall.smi','predictionoutput':'resultprediction.csv',
      'result':'result.txt','solvent':opts['solvent'],'tolerancec':str(opts['tolerancec']),
      'toleranceh':str(opts['toleranceh']),'spectruminput':'realspectrum.csv','clusteringoutput':'cluster.txt',
      'rberresolution':str(opts['rberresolution']),'louvainoutput':'clusterslouvain.txt',
      'usehsqctocsy':str(opts['usehsqctocsy']).lower(),'usehmbc':str(opts['usehmbc']).lower(),
      'dotwobonds':str(opts['dotwobonds']).lower(),'usedeeplearning':'false','debug':str(opts['debug']).lower(),
      'labelsimulated':str(opts['labelsimulated']).lower(),'generateplots':str(opts['generateplots']).lower(),'generatehtmlplots':str(opts.get('generatehtmlplots',True)).lower(),'htmlplotstopn':str(opts.get('htmlplotstopn',10)),'plotopacitymatched':str(opts.get('plotopacitymatched',0.95)),'plotopacitysimulated':str(opts.get('plotopacitysimulated',0.45)),'plotopacityunmatched':str(opts.get('plotopacityunmatched',0.28)),'plotopacitymisc':str(opts.get('plotopacitymisc',0.12)),'hmbcbruker':'NaN','hsqcbruker':'NaN','hsqctocsybruker':'NaN'}
    with open(project_dir/'nmrproc.properties','w',encoding='utf-8') as f: cp.write(f)

def global_props():
    # Java reads the root properties file before project overrides. Point it at Streamlit's project store.
    cp=configparser.ConfigParser(); cp.read(ROOT/'nmrproc.properties')
    cp['onesectiononly']['datadir']=java_property_path(PROJECTS)
    for k in ('hmbcbruker','hsqcbruker','hsqctocsybruker'): cp['onesectiononly'][k]='NaN'
    with open(ROOT/'nmrproc.properties','w',encoding='utf-8') as f: cp.write(f)

def run(cmd, log, timeout=900):
    p=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    log.append('$ '+' '.join(map(str,cmd))+'\n'+p.stdout)
    if p.returncode: raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(map(str,cmd))}\n{p.stdout[-4000:]}")
    return p.stdout

def run_live(cmd, log, status, timeout=1800):
    """Run a command while streaming its output into the Streamlit status panel."""
    started=time.monotonic()
    p=subprocess.Popen(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=1)
    lines=[]
    try:
        for line in iter(p.stdout.readline, ''):
            if line:
                line=line.rstrip()
                lines.append(line)
                status.write(line)
            if time.monotonic()-started > timeout:
                p.kill()
                raise TimeoutError(f"Stage 4 exceeded {timeout//60} minutes. Last output: " + (lines[-1] if lines else 'none'))
        rc=p.wait()
    finally:
        if p.stdout: p.stdout.close()
    output='\n'.join(lines)
    log.append('$ '+' '.join(map(str,cmd))+'\n'+output)
    if rc:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(map(str,cmd))}\n{output[-6000:]}")
    return output

def java_cp():
    sep=';' if os.name=='nt' else ':'
    return str(ROOT/'lib'/'simulate.jar')+sep+str(ROOT/'lib'/'*')

def validate(candidate, spectrum, names, unlabeled_type=''):
    errs=[]
    if not candidate: errs.append('Candidate structure file is required.')
    if not spectrum: errs.append('Measured spectrum file is required.')
    if candidate:
        txt=candidate.getvalue().decode('utf-8-sig',errors='replace')
        lines=[x.strip() for x in txt.splitlines() if x.strip()]
        if not lines: errs.append('Candidate file is empty.')
    if spectrum:
        txt=spectrum.getvalue().decode('utf-8-sig',errors='replace')
        records, skipped = parse_measured_spectrum_text(txt, default_type=unlabeled_type)
        if not records:
            errs.append('Measured spectrum contains no readable 13C/1H peak pairs. TAB, comma and semicolon delimiters are accepted.')
        else:
            untyped=sum(1 for _c,_h,t in records if not t)
            if untyped:
                errs.append(f'{untyped}/{len(records)} measured peaks have no experiment type. For a two-column peak list, select its experiment type in the input options; mixed experiments require section labels or a third type column.')
    return errs

def zip_project(pdir):
    bio=io.BytesIO()
    prediction_path = pdir / 'result' / 'resultprediction.csv'
    organized = None
    if (pdir / 'result' / 'atomic_predictions.csv').exists():
        export_bytes, _, _ = build_prediction_zip(pdir)
        organized = zipfile.ZipFile(io.BytesIO(export_bytes))
    with zipfile.ZipFile(bio,'w',zipfile.ZIP_DEFLATED) as z:
        for p in pdir.rglob('*'):
            if p.is_file():
                relative = p.relative_to(pdir)
                if organized is not None and p == prediction_path:
                    z.write(p, 'original/resultprediction.csv')
                    z.write(p, relative)
                else:
                    z.write(p, relative)
        if organized is not None:
            for filename in organized.namelist():
                z.writestr('calculated_NMR/' + filename, organized.read(filename))
    if organized is not None:
        organized.close()
    return bio.getvalue()

# Sidebar — branding, inputs and configuration
with st.sidebar:
    logo1 = ROOT / 'static' / 'NMRfilter_icon.png'
    logo2 = ROOT / 'static' / 'LAABio.png'
    if logo1.exists(): st.image(str(logo1), use_container_width=True)
    if logo2.exists(): st.image(str(logo2), use_container_width=True)

    st.header('NMRfilter')

    tutorial_url = 'https://github.com/RicardoMBorges/NMRfilter_Streamlit/blob/main/NMRfilter_Complete_Tutorial.md'
    tutorial_pt_url = 'https://github.com/RicardoMBorges/NMRfilter_Streamlit/blob/main/NMRfilter_Tutorial_Completo_pt.md'
    video_url = 'https://www.youtube.com/watch?v=pkY-rmvfDdU'
    mock_data_url = 'https://github.com/RicardoMBorges/NMRfilter_Streamlit/tree/main/mock_data'
    st.link_button('Tutorial-En', tutorial_url, use_container_width=True)
    st.link_button('Tutorial-Pt', tutorial_pt_url, use_container_width=True)
    st.link_button('Video', video_url, use_container_width=True)
    st.link_button('Mock data', mock_data_url, use_container_width=True)

    st.caption('Streamlit interface for the original NMRfilter v1.5 pipeline')

    with st.expander('Input format and workflow', expanded=False):
        st.markdown('**Candidates:** one SMILES per line (`.smi` or text). **Measured spectrum:** 13C and 1H shifts with experiment identity (HMBC/HSQC/HSQCTOCSY). TAB, comma and semicolon files are accepted; use legacy section labels or a third `type` column. Candidate names are optional but recommended for plots. The app runs the original conversion, simulation, clustering/community detection and similarity ranking pipeline.')

    st.subheader('Inputs')
    project_name=st.text_input('Project name','nmrfilter_run')
    candidate=st.file_uploader('Candidate structures — SMILES, one per line',type=['smi','txt','csv'])
    spectrum=st.file_uploader('Measured 2D NMR spectrum — 13C and 1H shifts',type=['csv','txt','tsv'])
    names=st.file_uploader('Candidate names (optional, one per candidate)',type=['txt','csv'])

    st.subheader('Parameters')
    solvent=st.selectbox('Solvent',SOLVENTS)
    c1,c2=st.columns(2)
    tolerancec=c1.number_input('13C tolerance (ppm)',0.001,10.0,0.2,0.01)
    toleranceh=c2.number_input('1H tolerance (ppm)',0.001,2.0,0.02,0.01,format='%.3f')
    rber=st.number_input('Louvain/RBER resolution',0.01,10.0,0.2,0.05)
    usehmbc=st.checkbox('Use HMBC',True)
    usehsqctocsy=st.checkbox('Use HSQC-TOCSY',False)
    dotwobonds=st.checkbox('Use 2 HOSE spheres',False)
    labels=st.checkbox('Label simulated spectra',False, help='Text label allocation can be slow for dense spectra.')
    generateplots=st.checkbox('Generate legacy PNG candidate plots',False, help='Optional legacy Matplotlib output. This is independent of the interactive HTML plots below.')
    generatehtmlplots=st.checkbox('Generate interactive HTML plots',True, help='Creates standalone Plotly HTML files after numerical ranking, without changing the matching calculation.')
    htmlplotstopn=st.number_input('Interactive plots — Top N candidates',1,30,10,1, help='HTML plots are generated only for the best-ranked N candidates to avoid unnecessary rendering.')
    with st.expander('Plot appearance', expanded=False):
        st.caption('Opacity of markers in the interactive HMBC/HSQC HTML plots.')
        plotopacitymatched=st.slider('Matched — solid green circles',0.0,1.0,0.95,0.05)
        plotopacitysimulated=st.slider('Simulated — gray open circles',0.0,1.0,0.45,0.05)
        plotopacityunmatched=st.slider('Unmatched — red open squares',0.0,1.0,0.28,0.05)
        plotopacitymisc=st.slider('Miscellaneous / unused — gray open squares',0.0,1.0,0.12,0.05)
        st.caption(f'Values used on next run: matched={plotopacitymatched:.2f} · simulated={plotopacitysimulated:.2f} · unmatched={plotopacityunmatched:.2f} · miscellaneous={plotopacitymisc:.2f}')
    debug=st.checkbox('Debug output',False)
    unlabeled_mode=st.selectbox(
        'Two-column spectrum interpretation',
        ['Reject as ambiguous','HMBC','HSQC','HSQC-TOCSY'],
        index=0,
        help='Used only for numeric rows that do not already carry an experiment label. For mixed HMBC/HSQC data, keep Reject and label sections/rows explicitly.'
    )
    unlabeled_type={'Reject as ambiguous':'','HMBC':'HMBC','HSQC':'HSQC','HSQC-TOCSY':'HSQCTOCSY'}[unlabeled_mode]

    st.divider()
    st.subheader('Contact')
    st.markdown('**Ricardo M Borges:** [ricardo_mborges@ufrj.br](mailto:ricardo_mborges@ufrj.br)  \n**Stefan Kuhn:** [stefan.kuhn@ut.ee](mailto:stefan.kuhn@ut.ee)')

    with st.expander('Cite', expanded=False):
        st.markdown("""
**NMRfilter**  
Kuhn, S., Colreavy-Donnelly, S., de Andrade Silva Quaresma, L. E. *et al.* (2020). Applying NMR compound identification using NMRfilter to match predicted to experimental data. *Metabolomics*, **16**, 123.  
[https://doi.org/10.1007/s11306-020-01748-1](https://doi.org/10.1007/s11306-020-01748-1)

**Integrated MS/NMR mixture analysis**  
Kuhn, S., Colreavy-Donnelly, S., de Souza, J. S., & Borges, R. M. (2019). An integrated approach for mixture analysis using MS and NMR techniques. *Faraday Discussions*, **218**, 339–353.

[PubMed record](https://pubmed.ncbi.nlm.nih.gov/33222074/)
        """)

st.title('NMRfilter')
workflow_tab, structure_tab = st.tabs(['Analysis and results', 'Structures and atomic shifts'])
with workflow_tab:
    st.caption('Run the NMRfilter workflow using the inputs and parameters in the sidebar.')

    if st.button('Run NMRfilter',type='primary',use_container_width=True):
        errs=validate(candidate,spectrum,names,unlabeled_type)
        if errs:
            for e in errs: st.error(e)
        else:
            pname=safe_name(project_name)+'_'+uuid.uuid4().hex[:8]
            pdir=PROJECTS/pname; pdir.mkdir(parents=True)
            (pdir/'testall.smi').write_bytes(candidate.getvalue())
            spectrum_text=spectrum.getvalue().decode('utf-8-sig',errors='replace')
            spectrum_records,_=parse_measured_spectrum_text(spectrum_text, default_type=unlabeled_type)
            if unlabeled_type:
                st.info(f'Unlabeled two-column peaks are being interpreted as {unlabeled_type}. Explicit labels in the file take precedence.')
            (pdir/'realspectrum.csv').write_text(to_legacy_tsv(spectrum_records),encoding='utf-8')
            if names: (pdir/'testallnames.txt').write_bytes(names.getvalue())
            opts=dict(solvent=solvent,tolerancec=tolerancec,toleranceh=toleranceh,rberresolution=rber,usehmbc=usehmbc,usehsqctocsy=usehsqctocsy,dotwobonds=dotwobonds,labelsimulated=labels,generateplots=generateplots,generatehtmlplots=generatehtmlplots,htmlplotstopn=int(htmlplotstopn),plotopacitymatched=plotopacitymatched,plotopacitysimulated=plotopacitysimulated,plotopacityunmatched=plotopacityunmatched,plotopacitymisc=plotopacitymisc,debug=debug)
            st.session_state['prediction_project'] = str(pdir)
            write_props(pdir,opts); global_props(); logs=[]
            status=st.status('Running NMRfilter…',expanded=True)
            try:
                status.write('1/4 Preparing project')
                run([sys.executable,'nmrfilter.py',pname],logs)
                status.write('Exporting complete atomic 13C/1H predictions from the HOSE simulator')
                run(['java','-cp',java_cp(),'uk.ac.dmu.simulate.AtomicPredictions',str(pdir.resolve())],logs)
                status.write('2/4 Converting candidate structures')
                out=run(['java','-cp',java_cp(),'uk.ac.dmu.simulate.Convert',pname],logs)
                # Respredict is deliberately unsupported in this wrapper; default engine uses HOSE prediction.
                status.write('3/4 Simulating candidate spectra')
                run(['java','-cp',java_cp(),'uk.ac.dmu.simulate.Simulate',pname],logs)
                status.write('4/4 Clustering measured peaks and ranking candidates')
                run_live([sys.executable,'-u','nmrfilter2.py',pname],logs,status,timeout=1800)
                status.update(label='NMRfilter completed',state='complete',expanded=False)
                result=pdir/'result'/'result.txt'
                if not result.exists(): result=pdir/'result.txt'
                st.success('Pipeline completed successfully.')
                ranking_tsv=pdir/'result'/'ranking_table.tsv'
                if ranking_tsv.exists():
                    st.subheader('Ranking')
                    rdf=pd.read_csv(ranking_tsv,sep='\t')
                    # Preserve matched/total text while exposing numeric percentages as progress bars.
                    rename={c:c.replace('Mathcing rate','Matching rate') for c in rdf.columns}
                    rdf=rdf.rename(columns=rename)
                    progress_cols=[]
                    for c in list(rdf.columns):
                        if c.startswith('Matching rate '):
                            vals=pd.to_numeric(rdf[c].astype(str).str.replace('%','',regex=False),errors='coerce')
                            rdf[c]=vals
                            progress_cols.append(c)
                    cfg={
                        'Rank': st.column_config.NumberColumn('Rank',format='%d'),
                        'Distance': st.column_config.NumberColumn('Distance',format='%.2f'),
                        'Standard deviation': st.column_config.NumberColumn('Standard deviation',format='%.2f'),
                    }
                    for c in progress_cols:
                        cfg[c]=st.column_config.ProgressColumn(c,help='Fraction of simulated correlations matched by the measured spectrum.',format='%.1f%%',min_value=0,max_value=100)
                    st.dataframe(rdf,use_container_width=True,hide_index=True,column_config=cfg)
                    st.download_button('Download ranking table (.tsv)',ranking_tsv.read_bytes(),file_name=f'{safe_name(project_name)}_ranking.tsv',mime='text/tab-separated-values')
                    with st.expander('Legacy text ranking',expanded=False):
                        if result.exists(): st.code(result.read_text(encoding='utf-8',errors='replace'))
                elif result.exists():
                    st.subheader('Ranking')
                    st.code(result.read_text(encoding='utf-8',errors='replace'))
                else: st.warning('Pipeline exited successfully, but ranking output was not found. Inspect the run log below.')
                html_dir=pdir/'plots_html'
                html_plots=sorted(html_dir.glob('*.html')) if html_dir.exists() else []
                if html_plots:
                    st.subheader('Interactive candidate plots')
                    st.caption('These HTML plots are generated from the same matched/unmatched arrays used by the numerical ranking.')
                    try:
                        import streamlit.components.v1 as components
                        for hp in html_plots:
                            with st.expander(hp.stem.replace('_',' '), expanded=False):
                                html_text=hp.read_text(encoding='utf-8',errors='replace')
                                components.html(html_text,height=650,scrolling=True)
                                st.download_button('Download HTML',html_text,file_name=hp.name,mime='text/html',key='html_'+hp.name)
                        html_zip=io.BytesIO()
                        with zipfile.ZipFile(html_zip,'w',zipfile.ZIP_DEFLATED) as z:
                            for hp in html_plots: z.write(hp,hp.name)
                        st.download_button('Download all interactive plots (.zip)',html_zip.getvalue(),file_name=f'{safe_name(project_name)}_interactive_plots.zip',mime='application/zip')
                    except Exception as plot_error:
                        st.warning(f'HTML plots were generated but could not be embedded: {plot_error}')
                plots=list((pdir/'plots').glob('*.png')) if (pdir/'plots').exists() else []
                if plots:
                    st.subheader('Legacy PNG plots')
                    for p in plots[:30]: st.image(str(p),caption=p.name)
                st.download_button('Download complete project/results ZIP',zip_project(pdir),file_name=f'{safe_name(project_name)}_nmrfilter_results.zip',mime='application/zip')
            except Exception as e:
                status.update(label='NMRfilter stopped',state='error',expanded=True)
                st.error(str(e))
            with st.expander('Run log',expanded=False): st.code('\n\n'.join(logs) if logs else 'No commands executed.')

    # Retain access across reruns and after a ranking failure.
    if st.session_state.get('prediction_project'):
        export_project = Path(st.session_state['prediction_project'])
        if (export_project / 'result' / 'atomic_predictions.csv').exists():
            st.subheader('Calculated NMR data')
            try:
                prediction_zip, candidate_count, correlation_count = build_prediction_zip(export_project)
                st.caption(f'{candidate_count} input entries · {correlation_count} calculated correlations. Complete atomic 13C/1H predictions, entry status, per-structure tables and original outputs are included.')
                st.download_button('Download calculated NMR data (.zip)', prediction_zip,
                    file_name=f'{export_project.name}_calculated_NMR.zip', mime='application/zip',
                    key='calculated_nmr_download')
            except (OSError, ValueError) as export_error:
                st.warning(f'Calculated NMR export unavailable: {export_error}')

    # Diagnose database coverage from the direct atomic simulator output.
    if st.session_state.get('prediction_project'):
        diagnostic_project = Path(st.session_state['prediction_project'])
        atomic_file = diagnostic_project / 'result' / 'atomic_predictions.csv'
        if atomic_file.exists():
            with st.expander('HOSE diagnostics — compounds with limited database support', expanded=False):
                st.markdown('HOSE sphere count describes how many layers of the local atomic environment matched the bundled reference database. Fewer spheres mean a less specific match; this is a screening flag, not a calibrated error estimate or proof of an incorrect structure.')
                threshold = st.selectbox('Flag predictions with fewer than this number of HOSE spheres', [4, 3, 2, 5, 6], key='hose_diagnostic_threshold', help='Default: flag 1–3 spheres. This threshold is a configurable screening choice, not a validated confidence cutoff.')
                nucleus_filter = st.selectbox('Inspect nucleus', ['Both', '13C', '1H'], key='hose_diagnostic_nucleus')
                try:
                    from hose_diagnostics import summarize_hose
                    atom_data = pd.read_csv(atomic_file, keep_default_na=False)
                    if nucleus_filter != 'Both':
                        atom_data = atom_data[atom_data['nucleus'] == nucleus_filter]
                    summary, flagged_atoms = summarize_hose(atom_data, int(threshold))
                    if summary.empty:
                        st.success('No low-sphere or missing atomic predictions for the selected nucleus and threshold.')
                    else:
                        st.caption(f'{len(summary)} compounds flagged. Counts refer to individual atoms, including equivalent hydrogens.')
                        st.dataframe(summary, use_container_width=True, hide_index=True)
                        st.download_button('Download compound diagnostic (.csv)', summary.to_csv(index=False).encode('utf-8-sig'), file_name='HOSE_compound_diagnostic.csv', mime='text/csv', key='hose_summary_download')
                        st.markdown('**Atoms requiring closer inspection**')
                        st.dataframe(flagged_atoms, use_container_width=True, hide_index=True)
                        st.download_button('Download flagged atoms (.csv)', flagged_atoms.to_csv(index=False).encode('utf-8-sig'), file_name='HOSE_flagged_atoms.csv', mime='text/csv', key='hose_atoms_download')
                    status_file = diagnostic_project / 'result' / 'atomic_prediction_status.csv'
                    if status_file.exists():
                        statuses = pd.read_csv(status_file, keep_default_na=False)
                        structure_issues = statuses[statuses['status'].isin(['structure_error', 'no_target_atoms', 'no_prediction'])]
                        if not structure_issues.empty:
                            st.markdown('**Entries without usable predictions**')
                            st.dataframe(structure_issues, use_container_width=True, hide_index=True)
                    st.markdown('**Help improve the reference database.** For flagged compounds, seek reliable experimental ¹³C/¹H assignments in your own measurements or the literature. Once the structure and assignments are verified, consider submitting the experimental data to nmrshiftdb2 with solvent, acquisition conditions and source references. Public submissions become available after reviewer approval. The simulated shifts shown here should not be presented as experimental measurements.')
                    st.link_button('Open nmrshiftdb2', 'https://nmrshiftdb.nmr.uni-koeln.de/')
                    st.link_button('Submission and review instructions', 'https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/using.html')
                    st.caption('The app uses a bundled database snapshot. A submission to the live database does not update the local HOSE tables automatically; low support does not prove that data are absent from the current online database.')
                except (OSError, ValueError, KeyError) as diagnostic_error:
                    st.warning(f'HOSE diagnostic unavailable: {diagnostic_error}')

with structure_tab:
    from structure_view import render_structure_tab
    render_structure_tab(st.session_state.get('prediction_project'))
