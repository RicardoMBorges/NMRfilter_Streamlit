"""Integration checks: direct simulator output, completeness, identity and raw preservation."""
import csv
import io
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from prediction_export import build_prediction_zip

ROOT = Path(__file__).resolve().parent
JAVA = shutil.which('java') or '/usr/lib/jvm/java-17-openjdk-amd64/bin/java'

class AtomicExportTests(unittest.TestCase):
    def test_full_atoms_and_legacy_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); project = base/'valid'; project.mkdir()
            props = '[onesectiononly]\ndatadir=.\nmsmsinput=testall.smi\npredictionoutput=resultprediction.csv\nsolvent=Unreported\nusehmbc=true\nusehsqctocsy=false\ndotwobonds=false\nusedeeplearning=false\ndebug=false\n'
            (base/'nmrproc.properties').write_text(props)
            (project/'nmrproc.properties').write_text(props)
            (project/'testall.smi').write_text('CCO\nCC(=O)O\n')
            (project/'testallnames.txt').write_text('ethanol\nacetic acid\n')
            cp = str(ROOT/'lib'/'*')
            subprocess.run([JAVA,'-cp',cp,'uk.ac.dmu.simulate.AtomicPredictions',str(project)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            subprocess.run([JAVA,'-cp',cp,'uk.ac.dmu.simulate.Simulate','valid'],cwd=base,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            atom_rows = list(csv.DictReader(io.StringIO((project/'result/atomic_predictions.csv').read_text())))
            self.assertEqual(len(atom_rows),14) # C2H6 + C2H4; includes carbonyl and OH.
            lookup={(int(r['candidate_id']),r['nucleus'],int(r['atom_index'])):float(r['shift_ppm']) for r in atom_rows}
            candidate=1
            raw=(project/'result/resultprediction.csv').read_bytes()
            for line in raw.decode().splitlines():
                if not line.strip(): continue
                if line.strip()=='/': candidate+=1; continue
                c,h,experiment,ci,hi = next(csv.reader([line]))
                self.assertEqual(float(c),lookup[candidate,'13C',int(ci)])
                self.assertEqual(float(h),lookup[candidate,'1H',int(hi)])
            data, n, correlations=build_prediction_zip(project)
            self.assertEqual(n,2); self.assertGreater(correlations,0)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                self.assertEqual(z.read('original/resultprediction.csv'),raw)
                rows=list(csv.DictReader(io.StringIO(z.read('resultprediction.csv').decode('utf-8-sig'))))
                self.assertEqual(len(rows),14)
                self.assertTrue(any(r['candidate_id']=='2' and r['nucleus']=='13C' and float(r['shift_ppm'])>100 for r in rows))
                self.assertIn('candidates/0002_acetic_acid/1H.csv',z.namelist())

    def test_invalid_entry_does_not_shift_following_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'nmrproc.properties').write_text('[onesectiononly]\nmsmsinput=testall.smi\nsolvent=Unreported\nusedeeplearning=false\n')
            (p/'testall.smi').write_text('CCO first\n\nC(C invalid\n[Na+] sodium\nCCO duplicate\n')
            subprocess.run([JAVA,'-cp',str(ROOT/'lib/*'),'uk.ac.dmu.simulate.AtomicPredictions',str(p)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            statuses=list(csv.DictReader(io.StringIO((p/'result/atomic_prediction_status.csv').read_text())))
            self.assertEqual([r['candidate_id'] for r in statuses],['1','2','3','4'])
            self.assertEqual(statuses[1]['status'],'structure_error')
            self.assertEqual(statuses[2]['status'],'no_target_atoms')
            self.assertEqual(statuses[3]['candidate_name'],'duplicate')
            data,n,_=build_prediction_zip(p) # also available if 2D/ranking fails
            self.assertEqual(n,4)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                missing=list(csv.DictReader(io.StringIO(z.read('entries_without_prediction.csv').decode('utf-8-sig'))))
                self.assertEqual([r['candidate_id'] for r in missing],['2','3'])
                self.assertEqual(z.read('candidates/0002_invalid/13C.csv').decode('utf-8-sig').count('\n'),1)

if __name__=='__main__': unittest.main()
