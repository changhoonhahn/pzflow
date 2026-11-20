"""Validate delivered numerical statements, notebook execution and PDF structure."""
from pathlib import Path
import json
import re
import sys
import numpy as np
import nbformat
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from .paper_utils import load_predictions,load_metrics,summary

def main():
    paper=ROOT/'paper';d=load_predictions(ROOT);m=load_metrics(ROOT)
    names=['cosmos_pdf_flow','broadband_spec_flow','eazy_rprior','lephare_full_cosmos']
    for name in names:
        expected=(m['flow']['models'][name]['all'] if name=='cosmos_pdf_matched_flow'
                  else m['templates']['models'][name])
        result=summary(d['zspec'],d[f'{name}_quantiles'])
        for key,value in result.items():
            assert np.isclose(value,expected[key],atol=1e-8),(name,key,value,expected[key])
    sources=['main.tex','intro.tex','methods.tex','bootstrapping.tex','appendix.tex','results.tex','discussion.tex','table_results.tex']
    tex='\n'.join((paper/p).read_text() for p in sources)
    cites=set()
    for group in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}',tex):
        cites.update(k.strip() for k in group.split(','))
    bibkeys=set(re.findall(r'@\w+\{([^,]+),',(paper/'references.bib').read_text()))
    assert cites<=bibkeys, cites-bibkeys
    log=(paper/'main.log').read_text()
    assert 'undefined' not in log.lower()
    assert 'Overfull' not in log
    reader=PdfReader(paper/'main.pdf')
    assert not reader.is_encrypted and len(reader.pages)>=1
    assert not (reader.metadata.author or '').strip()
    text='\n'.join(page.extract_text() or '' for page in reader.pages)
    compact=re.sub(r'\s+','',text)
    assert '111,952' in text and '5,083' in text and 'Table1.' in compact
    assert tex.count(r'\begin{figure*}')==2
    assert tex.count(r'\begin{figure}')==3
    assert tex.count(r'\begin{deluxetable*}')==1
    for n in range(1,5):assert f'Figure{n}.' in compact
    assert 'Figure5.' in compact or 'FigureA1.' in compact
    for excluded in ['COSMOS PDFs, matched objects','EAZY, flat prior','Spectroscopic labels']:
        assert excluded not in tex
    checks=[]
    for path in sorted((ROOT/'nb').glob('figure_0[1-4]_*.ipynb')):
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        code=[c for c in nb.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code)
        outputs=[o for c in code for o in c.outputs]
        assert all(o.output_type!='error' for o in outputs)
        assert any('image/png' in o.get('data',{}) for o in outputs)
        assert any('save_figure' in c.source for c in code)
        checks.append(path.name)
    assert len(checks)==4
    result=dict(n_test=len(d['id']),numerical_models_verified=names,
                bibliography_keys_verified=sorted(cites),pdf_pages=len(reader.pages),
                overfull_boxes=0,unresolved_references=0,author_metadata_unset=True,
                main_figures=4,appendix_figures=1,main_tables=1,executed_notebooks=checks)
    (paper/'paper_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
