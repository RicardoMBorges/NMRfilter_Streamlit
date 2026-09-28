from pathlib import Path
import re
from plotly.subplots import make_subplots
import plotly.graph_objects as go

EXP = [('HMBC',0),('HSQC',1),('HSQC-TOCSY',2)]

def _safe(s):
    return re.sub(r'[^A-Za-z0-9_.-]+','_',str(s)).strip('._') or 'candidate'

def _unused(real, matched_x, matched_y, near_x, near_y, idx):
    used={(float(x),float(y)) for x,y in zip(matched_x[idx],matched_y[idx])}
    used.update((float(x),float(y)) for x,y in zip(near_x[idx],near_y[idx]))
    out=[]
    for c,h,t in real:
        is_type = (idx==0 and 'HMBC' in t) or (idx==1 and 'HSQC' in t and 'TOCSY' not in t) or (idx==2 and 'HSQCTOCSY' in t)
        if is_type and (float(c),float(h)) not in used: out.append((float(c),float(h)))
    return out

def generate_interactive_html_plots(project_dir, ranked_positions, names, smiles, costs_norm, std_norm,
                                    spectrum_real, xsim, ysim, xreal, yreal, xnear, ynear,
                                    usehmbc=True, usehsqctocsy=False, top_n=10, experiment_available=None, plot_opacity=None):
    experiment_available = experiment_available or {'HMBC':True,'HSQC':True,'HSQC-TOCSY':True}
    plot_opacity = plot_opacity or {'matched':0.95,'simulated':0.45,'unmatched':0.28,'misc':0.12}
    outdir=Path(project_dir)/'plots_html'; outdir.mkdir(parents=True,exist_ok=True)
    for old in outdir.glob('*.html'): old.unlink()
    active=[]
    if usehmbc: active.append(('HMBC',0))
    active.append(('HSQC',1))
    if usehsqctocsy: active.append(('HSQC-TOCSY',2))
    written=[]
    for rank,pos in enumerate(ranked_positions[:int(top_n)],1):
        name=names[pos] if pos < len(names) else f'candidate_{pos+1}'
        title=f'Rank {rank} — {name} | distance={costs_norm.get(pos,float("nan")):.2f}'
        if pos in std_norm: title += f' | normalized SD={std_norm[pos]:.2f}'
        fig=make_subplots(rows=1,cols=len(active),subplot_titles=[a[0] for a in active],shared_yaxes=True,horizontal_spacing=0.07)
        for col,(label,idx) in enumerate(active,1):
            sim=list(zip(xsim[pos][idx],ysim[pos][idx])); matched=list(zip(xreal[pos][idx],yreal[pos][idx])); near=list(zip(xnear[pos][idx],ynear[pos][idx])); unused=_unused(spectrum_real,xreal[pos],yreal[pos],xnear[pos],ynear[pos],idx)
            def add(points,name,symbol,size,opacity,rgb,line_width=1.0,open_marker=False,showlegend=True):
                if not points:return
                cs=[p[0] for p in points]; hs=[p[1] for p in points]
                opacity=max(0.0,min(1.0,float(opacity)))
                r,g,b=rgb
                rgba=f'rgba({r},{g},{b},{opacity:.3f})'
                if open_marker:
                    marker=dict(symbol=symbol,size=size,opacity=1.0,color=rgba,line=dict(color=rgba,width=line_width))
                else:
                    marker=dict(symbol=symbol,size=size,opacity=1.0,color=rgba,line=dict(color=rgba,width=line_width))
                fig.add_trace(go.Scatter(x=hs,y=cs,mode='markers',name=name,showlegend=showlegend,legendgroup=name,marker=marker,customdata=[[c,h,label,name] for c,h in points],hovertemplate='δ1H=%{x:.3f} ppm<br>δ13C=%{y:.3f} ppm<br>Experiment=%{customdata[2]}<br>Status=%{customdata[3]}<extra></extra>'),row=1,col=col)
            # Visual hierarchy: simulated = gray open circles; matches = solid green circles;
            # experimental non-matches = red open squares; unused/miscellaneous = gray open squares.
            add(sim,'Simulated','circle-open',5,plot_opacity['simulated'],(105,105,105),1.1,True,col == 1)
            add(matched,'Matched','circle',5,plot_opacity['matched'],(34,139,34),0.6,False,col == 1)
            add(near,'Unmatched','square-open',4,plot_opacity['unmatched'],(190,45,45),1.0,True,col == 1)
            add(unused,'Unused','square-open',4,plot_opacity['misc'],(120,120,120),0.9,True,col == 1)
            # A real frame around each spectrum; mirror keeps all four sides visible and stable on resize/zoom.
            fig.update_xaxes(title_text='δ1H (ppm)',autorange='reversed',range=[10,0],showline=True,mirror=True,linecolor='rgba(70,70,70,0.75)',linewidth=1,row=1,col=col)
            fig.update_yaxes(title_text='δ13C (ppm)' if col == 1 else None,autorange='reversed',range=[200,0],showline=True,mirror=True,linecolor='rgba(70,70,70,0.75)',linewidth=1,row=1,col=col)
            if not experiment_available.get(label, False):
                fig.add_annotation(text=f'<b>{label} — Not evaluated</b><br>No experimental {label} spectrum was provided.<br>{len(sim)} correlations predicted for this candidate.', xref='x domain', yref='y domain', x=0.5, y=0.5, showarrow=False, align='center', bgcolor='rgba(255,255,255,0.85)', bordercolor='rgba(120,120,120,0.5)', row=1, col=col)
        rates=[]
        for label,idx in active:
            den=len(ysim[pos][idx]); num=len(yreal[pos][idx])
            if not experiment_available.get(label, False): rates.append(f'{label} N/A (not evaluated)')
            else: rates.append(f'{label} {num}/{den}' + (f' ({num/den:.1%})' if den else ''))
        opacity_note = 'Opacity used — matched {matched:.2f}; simulated {simulated:.2f}; unmatched {unmatched:.2f}; miscellaneous {misc:.2f}'.format(**plot_opacity)
        fig.update_layout(title=dict(text=title+' | '+'; '.join(rates)+'<br><sup>'+opacity_note+'</sup>',y=0.985,yanchor='top'),height=650,template='plotly_white',hovermode='closest',legend=dict(orientation='h',yanchor='bottom',y=1.10,xanchor='center',x=0.5,font=dict(size=10),itemsizing='constant',title_text=''),margin=dict(t=205))
        fig.add_annotation(text=f'SMILES: {smiles[pos] if pos < len(smiles) else ""}',xref='paper',yref='paper',x=0,y=-0.08,showarrow=False,align='left',font=dict(size=10))
        path=outdir/f'rank_{rank:02d}_{_safe(name)}.html'
        fig.write_html(path,include_plotlyjs=True,full_html=True,config={'displaylogo':False,'scrollZoom':True,'toImageButtonOptions':{'format':'svg'}})
        written.append(path)
    return written
