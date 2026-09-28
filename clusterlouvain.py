from igraph import *
import csv
# The original NMRfilter imported the legacy `louvain` package.
# Modern installations use `leidenalg`, which exposes the compatible
# find_partition/RBERVertexPartition API and ships wheels for current Python.
try:
    import leidenalg as louvain
except ImportError as exc:
    raise ImportError(
        "NMRfilter requires the 'leidenalg' package for community detection. "
        "Install the dependencies from requirements.txt."
    ) from exc
import os

def Two_Column_List_l(file):
    from spectrum_io import parse_measured_spectrum_file
    records, _ = parse_measured_spectrum_file(file)
    return [[i, c, h, typ] for i, (c, h, typ) in enumerate(records)]


def cluster2dspectrumlouvain(cp, project):
	datapath=cp.get('datadir')

	realpeaks = Two_Column_List_l(datapath+os.sep+project+os.sep+cp.get('spectruminput'))
	#print(realpeaks)
	g=Graph.Read_Edgelist(datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('clusteringoutput'),directed=False)
	#print(g)
	louvainresult= louvain.find_partition(g, louvain.RBERVertexPartition, resolution_parameter=float(cp.get('rberresolution')))
	#print(louvainresult)
	f=open(datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('louvainoutput'),'w')
	i=0
	for cluster in louvainresult:
		if len(cluster)>0:
			fsmarts=open(datapath+os.sep+project+os.sep+'result'+os.sep+'smart'+os.sep+'smart'+str(i)+'.csv','w')
			fsmarts.write('13C,1H\n')
			f.write('/\n')	
			for peak in cluster:
				for realpeak in realpeaks:
					if realpeak[0]==peak:
						f.write(str(realpeak[0])+','+str(realpeak[1])+','+str(realpeak[2])+'\n')
						if("HSQC" in realpeak[3] and not "TOCSY" in realpeak[3]):
							fsmarts.write(str(realpeak[1])+','+str(realpeak[2])+'\n')
			i += 1
	f.close()
