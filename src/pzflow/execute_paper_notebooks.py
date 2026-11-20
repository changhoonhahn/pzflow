"""Execute every paper notebook in a fresh Python kernel and verify its outputs."""
from pathlib import Path
import json
import os
import sys
import argparse
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostics',action='store_true',help='Execute the three standalone prior-correction notebooks instead of the four manuscript notebooks.')
    parser.add_argument('--stress',action='store_true',help='Execute the catastrophic-PDF injection figure notebook.')
    parser.add_argument('--all',action='store_true',help='Execute all eight notebooks in nb/.')
    parser.add_argument('--notebook',help='Execute only this notebook filename in the selected folder.')
    args=parser.parse_args()
    temp=ROOT/'tmp/jupyter';kernel=temp/'kernels/cosmos-paper';kernel.mkdir(parents=True,exist_ok=True)
    spec=dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
              display_name='COSMOS paper execution',language='python')
    (kernel/'kernel.json').write_text(json.dumps(spec))
    os.environ['JUPYTER_PATH']=str(temp)
    os.environ['IPYTHONDIR']=str(temp/'ipython')
    os.environ['JUPYTER_RUNTIME_DIR']=str(temp/'runtime')
    os.environ.setdefault('MPLCONFIGDIR',str(temp/'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME',str(temp/'cache'))
    for key in ['IPYTHONDIR','JUPYTER_RUNTIME_DIR','MPLCONFIGDIR','XDG_CACHE_HOME']:
        Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    checks=[]
    assert sum([args.diagnostics,args.stress,args.all])<=1
    folder=ROOT/'nb'
    prior_names={'figure_all_models.ipynb','figure_nz_ratio.ipynb','figure_prior_correction.ipynb'}
    for path in sorted(folder.glob('figure_*.ipynb')):
        if not args.all and args.diagnostics and path.name not in prior_names:continue
        if not args.all and args.stress and path.name!='figure_catastrophic_pdf_stress.ipynb':continue
        if not (args.all or args.diagnostics or args.stress) and not path.name.startswith(('figure_01_','figure_02_','figure_03_','figure_04_')):continue
        if args.notebook and path.name!=args.notebook:continue
        print(f'Executing {path.name}',flush=True)
        nb=nbformat.read(path,as_version=4)
        NotebookClient(nb,timeout=300,kernel_name='cosmos-paper',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbformat.validate(nb)
        errors=[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
        assert not errors
        displays=sum('image/png' in o.get('data',{}) for c in nb.cells if c.cell_type=='code' for o in c.outputs)
        assert displays>=1
        nbformat.write(nb,path)
        checks.append(dict(notebook=path.name,code_cells=sum(c.cell_type=='code' for c in nb.cells),inline_figures=displays,errors=0))
        print(f'Passed {path.name}',flush=True)
    assert len(checks)==(1 if (args.notebook or args.stress) else (8 if args.all else 3 if args.diagnostics else 4))
    dest=(ROOT/'results/notebook_validation.json' if args.all else
          ROOT/'results/catastrophic_pdf_stress/notebook_validation.json' if args.stress else
          ROOT/'results/coverage_prior_correction/notebook_validation.json' if args.diagnostics else
          ROOT/'paper/notebook_validation.json')
    if args.notebook and dest.exists():
        checks=sorted([c for c in json.loads(dest.read_text()) if c['notebook']!=args.notebook]+checks,
                      key=lambda c:c['notebook'])
    dest.write_text(json.dumps(checks,indent=2)+'\n')
    if args.all:
        groups=[(ROOT/'paper/notebook_validation.json',[c for c in checks if c['notebook'].startswith(('figure_01_','figure_02_','figure_03_','figure_04_'))]),
                (ROOT/'results/coverage_prior_correction/notebook_validation.json',[c for c in checks if c['notebook'] in prior_names]),
                (ROOT/'results/catastrophic_pdf_stress/notebook_validation.json',[c for c in checks if c['notebook']=='figure_catastrophic_pdf_stress.ipynb'])]
        for target,values in groups:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(values,indent=2)+'\n')

if __name__=='__main__':main()
