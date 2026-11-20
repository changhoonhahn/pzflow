"""Validate the portable source layout and versioned figure inputs, without training."""
from pathlib import Path
import ast
import json
import re
import numpy as np
import nbformat
from .paper_utils import load_predictions, load_metrics, summary

ROOT=Path(__file__).resolve().parents[2]


def main():
    files=list((ROOT/'src/pzflow').glob('*.py'))+list((ROOT/'bin').glob('*.py'))
    for path in files:ast.parse(path.read_text(),filename=str(path))
    for path in (ROOT/'bin').glob('*.py'):
        assert (ROOT/'src/pzflow'/path.name).exists(),path
        assert 'runpy.run_module' in path.read_text(),path
    notebooks=sorted((ROOT/'nb').glob('*.ipynb'))
    assert len(notebooks)==8
    for path in notebooks:
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        cells=[c for c in nb.cells if c.cell_type=='code']
        assert cells and all(c.execution_count is not None for c in cells),path
        outputs=[o for c in cells for o in c.outputs]
        assert not any(o.output_type=='error' for o in outputs),path
        assert any('image/png' in o.get('data',{}) for o in outputs),path
        for c in cells:
            assert not re.search(r'from (paper_utils|project_plot_style) import',c.source),path
            assert "ROOT/'notebooks'" not in c.source and "ROOT / 'notebooks'" not in c.source,path
    d=load_predictions(ROOT);m=load_metrics(ROOT)
    for name in ['cosmos_pdf_flow','broadband_spec_flow','eazy_rprior','lephare_full_cosmos']:
        actual=summary(d['zspec'],d[f'{name}_quantiles'])
        expected=m['templates']['models'][name]
        for key,value in actual.items():assert np.isclose(value,expected[key],atol=1e-8),(name,key)
    with np.load(ROOT/'results/catastrophic_pdf_stress/figure_inputs.npz') as stress:
        np.testing.assert_array_equal(stress['id'],d['id'])
    with np.load(ROOT/'results/coverage_prior_correction/all_models_diagnostics.npz') as prior:
        np.testing.assert_array_equal(prior['id'],d['id'])
    result=dict(source_files=len(files),entry_points=len(list((ROOT/'bin').glob('*.py'))),
                executed_notebooks=[p.name for p in notebooks],common_test_objects=len(d['id']),
                compact_predictions_verified=True,model_training_performed=False)
    dest=ROOT/'results/repository_validation.json'
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
