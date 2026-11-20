"""Package the reviewed manuscript, figure notebooks, compact data and analysis sources."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]

def main():
    paper=ROOT/'paper';output=ROOT/'output';(output/'pdf').mkdir(parents=True,exist_ok=True)
    pdf=output/'pdf/cosmos_posterior_transfer_apjl.pdf'
    shutil.copyfile(paper/'main.pdf',pdf)
    allowed={'.tex','.bib','.bst','.cls','.bbl'}
    files=[p for p in paper.iterdir() if p.is_file() and
           (p.suffix in allowed or p.name in ['main.pdf','Makefile','README.txt',
               'paper_validation.json','notebook_validation.json','word_count.txt'])]
    files+=list((paper/'data').glob('*'))+list((paper/'figures').glob('*'))
    files += [p for p in (ROOT/'nb').iterdir() if p.is_file() and p.suffix in ['.py','.ipynb','.txt']]
    files += [p for p in (ROOT/'bin').iterdir() if p.is_file() and p.suffix in ['.py','.txt']]
    files += list((ROOT/'src').glob('**/*.py'))+list((ROOT/'requirements').glob('*.txt'))
    files += [p for p in (ROOT/'docs').iterdir() if p.is_file()]
    files += [ROOT/n for n in ['README.md','pyproject.toml','Makefile','LICENSE']]
    diagnostics=ROOT/'results/coverage_prior_correction'
    compact={'diagnostics.npz','all_models_diagnostics.npz'}
    files += [p for p in diagnostics.iterdir() if p.is_file() and
              (p.suffix in ['.ipynb','.json','.png'] or p.name in compact)]
    files += list((ROOT/'results').glob('*/*.png'))+list((ROOT/'results').glob('*/*.pdf'))
    stress=ROOT/'results/catastrophic_pdf_stress'
    files += [p for p in stress.iterdir() if p.is_file() and
              (p.suffix in ['.json','.csv'] or p.name=='figure_inputs.npz')]
    files=sorted(set(p for p in files if p.is_file()))
    records={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
    manifest=paper/'package_manifest.json'
    manifest.write_text(json.dumps(dict(
        description='ApJL-style draft; authors unset; compact figure reproduction package',
        files=records,
        excluded=['paper/reference','raw catalogs','model checkpoints','TeX build intermediates']
    ),indent=2)+'\n')
    files.append(manifest)
    archive=output/'cosmos_posterior_transfer_apjl.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,'cosmos_posterior_transfer_apjl/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert not any('/reference/' in n or '__pycache__' in n for n in names)
        assert sum(n.endswith('.ipynb') for n in names)==8
        for path,meta in records.items():
            assert hashlib.sha256(z.read('cosmos_posterior_transfer_apjl/'+path)).hexdigest()==meta['sha256']
    print(json.dumps(dict(pdf=str(pdf),archive=str(archive),files=len(files),archive_bytes=archive.stat().st_size),indent=2))

if __name__=='__main__':main()
