import os
import sys
import time
from clustering import *
from clusterlouvain import *
from similarity import *
from nmrutil import *
from plotutil import *

def stage(label, func, *args):
    t=time.perf_counter()
    print(f"[NMRfilter] START {label}", flush=True)
    out=func(*args)
    print(f"[NMRfilter] DONE  {label} ({time.perf_counter()-t:.2f} s)", flush=True)
    return out

project=sys.argv[1]
cp=readprops(project)
datapath=cp.get('datadir')
print(f"[NMRfilter] project={project}", flush=True)
print(f"[NMRfilter] datadir={datapath}", flush=True)
stage('4.1 clustering measured spectrum', cluster2dspectrum, cp, project)
stage('4.2 community detection', cluster2dspectrumlouvain, cp, project)
stage('4.3 background preparation', generateBackgrounds, cp, project)
stage('4.4 similarity/ranking', similarity, cp, project, True)
print('[NMRfilter] Stage 4 completed.', flush=True)
