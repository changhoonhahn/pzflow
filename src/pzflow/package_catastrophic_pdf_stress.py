"""Deliver the completed injection experiment, figure notebook and checkpoints."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/catastrophic_pdf_stress'

def main():
    complete=json.loads((OUT/'completion.json').read_text())
    assert complete['completed'] and len(complete['scenarios'])==13
    checks=json.loads((OUT/'notebook_validation.json').read_text())
    assert len(checks)==1 and checks[0]['errors']==0
    original=ROOT/'results/flow_magnitude_limited/cosmos_pdf_flow.pt'
    source_manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256(original.read_bytes()).hexdigest()==source_manifest['baseline_checkpoint_sha256']
    shutil.copyfile(original,OUT/'baseline.pt')
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!='package_manifest.json'
           and p.suffix in ['.json','.npz','.pt','.png','.ipynb','.csv','.log']]
    files += list((ROOT/'bin').glob('*.py'))+list((ROOT/'src').glob('**/*.py'))
    files += list((ROOT/'requirements').glob('*.txt'))
    files += [ROOT/'nb/figure_catastrophic_pdf_stress.ipynb']
    files += [ROOT/'docs'/n for n in ['PLOTTING_STYLE.md','CATASTROPHIC_PDF_STRESS.txt']]
    files += [ROOT/n for n in ['README.md','pyproject.toml','LICENSE']]
    # Root marker for the notebook's portable project-root discovery.
    files += [ROOT/'paper/data/predictions.npz']
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)}
    (OUT/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files.append(OUT/'package_manifest.json')
    archive=ROOT/'output/catastrophic_pdf_stress.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,'catastrophic_pdf_stress/'+str(p.relative_to(ROOT)))
        z.writestr('catastrophic_pdf_stress/README.txt',
            'Completed random teacher-PDF corruption experiment: 12 retrained models plus one original baseline.\n'
            'Open nb/figure_catastrophic_pdf_stress.ipynb to regenerate figures.\n'
            'Install requirements/notebooks.txt and TeX (latex and dvipng); plotting needs no model retraining.\n'
            'The original training catalog is not included; full retraining requires the existing\n'
            'data/flow_magnitude_limited/prepared.npz snapshot. Model checkpoints are locally generated.\n'
            'Scientific limitations and the exact corruption model are in manifest.json and\n'
            'docs/CATASTROPHIC_PDF_STRESS.txt. Results use one training seed and one random pattern.\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p,m in manifest.items():assert hashlib.sha256(z.read('catastrophic_pdf_stress/'+p)).hexdigest()==m['sha256']
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size),indent=2))

if __name__=='__main__':main()
