"""Freeze small, portable figure inputs and generate manuscript numbers from saved results."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'paper'

def main():
    dest=PAPER/'data';dest.mkdir(parents=True,exist_ok=True)
    paths=['data/flow_magnitude_limited/prepared.npz',
           'results/flow_magnitude_limited/heldout_predictions.npz',
           'results/flow_magnitude_limited/comparison_metrics.json',
           'results/flow_magnitude_limited/selection_audit.json',
           'results/template_comparison/heldout_predictions.npz',
           'results/template_comparison/comparison_metrics.json',
           'results/template_comparison/validation.json',
           'results/cosmos2020_coverage/coverage_summary.json']
    source={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    d=np.load(ROOT/paths[0])
    np.savez_compressed(dest/'selection.npz',**{k:d[k] for k in ['id','imag','split','cell','zspec']})
    a=np.load(ROOT/paths[1]);t=np.load(ROOT/paths[4])
    np.testing.assert_array_equal(a['id'],t['id']);np.testing.assert_allclose(a['zspec'],t['zspec'])
    arrays={k:t[k] for k in t.files}
    for name in ['cosmos_pdf_flow','cosmos_pdf_matched_flow','broadband_spec_flow']:
        for suffix in ['quantiles','pit']:arrays[f'{name}_{suffix}']=a[f'{name}_{suffix}']
    np.savez_compressed(dest/'predictions.npz',**arrays)
    flow=json.loads((ROOT/paths[2]).read_text());selection=json.loads((ROOT/paths[3]).read_text())
    template=json.loads((ROOT/paths[5]).read_text());audit=json.loads((ROOT/paths[7]).read_text())
    def portable(value):
        if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
        if isinstance(value,list):return [portable(v) for v in value]
        if isinstance(value,str):return value.replace(str(ROOT)+'/', '')
        return value
    metrics=portable(dict(flow=flow,selection=selection,templates=template,teacher_audit=audit))
    (dest/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    manifest=dict(source_sha256=source,description='Portable figure inputs; no model fitting or resampling of the training population.',
                  source_population='Revised magnitude-limited experiment only; original oversampled experiment is excluded.',
                  column_notes={'imag':'native HSC i AB, 2 arcsec aperture, before noise',
                                'split':'0=train, 1=validation, 2=test, disjoint sky cells',
                                'zspec':'NaN if no qualifying public secure spectrum',
                                'quantiles':'0.025,0.16,0.50,0.84,0.975',
                                'pit':'CDF at the spectroscopic redshift'})
    (dest/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    models=template['models'].copy()
    models['cosmos_pdf_matched_flow']=flow['models']['cosmos_pdf_matched_flow']['all']
    names=['cosmos_pdf_flow','broadband_spec_flow','eazy_rprior','lephare_full_cosmos']
    labels=['Bootstrapped flow','Broadband flow','EAZY, supplied R prior','LePhare, full COSMOS']
    counts=['111,952','22,604',r'\nodata',r'\nodata']
    lines=[r'\begin{deluxetable*}{lrrrrrr}',r'\tablecaption{Performance on the same 5,083 held-out spectroscopic galaxies.\label{tab:results}}',
           r'\tablewidth{0pt}',r'\tablehead{\colhead{Method} & \colhead{$N_{\rm train}$} & \colhead{$b$} & \colhead{$\sigma_{\rm NMAD}$} & \colhead{$\eta$ (\%)} & \colhead{$C_{68}$ (\%)} & \colhead{$C_{95}$ (\%)}}',r'\startdata']
    for name,label,count in zip(names,labels,counts):
        m=models[name]
        lines.append(f"{label} & {count} & {m['bias']:+.4f} & {m['sigma_nmad']:.4f} & {100*m['outlier_fraction']:.2f} & {100*m['coverage68']:.1f} & {100*m['coverage95']:.1f}"+r' \\')
    lines.extend([r'\enddata',r'\tablecomments{All point estimates are posterior medians; $C_\alpha$ is equal-tailed interval coverage. The first three methods use six broadband measurements at inference; released LePhare uses the full deep COSMOS photometry and is the training teacher. EAZY is not calibrated on this population. An ellipsis indicates no training set fitted in this experiment, not absence of prior calibration in the released methods.}',r'\end{deluxetable*}'])
    (PAPER/'table_results.tex').write_text('\n'.join(lines)+'\n')
    f=models['cosmos_pdf_flow'];s=models['broadband_spec_flow'];e=models['eazy_rprior']
    values={'NParent':f"{len(d['id']):,}",'NTrain':f"{np.sum(d['split']==0):,}",'NValidation':f"{np.sum(d['split']==1):,}",
            'NPhotTest':f"{np.sum(d['split']==2):,}",'NTest':f"{len(a['id']):,}",
            'FlowScatter':f"{f['sigma_nmad']:.4f}",'SpecScatter':f"{s['sigma_nmad']:.4f}",
            'MatchedScatter':f"{models['cosmos_pdf_matched_flow']['sigma_nmad']:.4f}",
            'ScatterGainSpec':f"{100*(1-f['sigma_nmad']/s['sigma_nmad']):.0f}",
            'ScatterGainEazy':f"{100*(1-f['sigma_nmad']/e['sigma_nmad']):.0f}",
            'OutlierGainEazy':f"{100*(1-f['outlier_fraction']/e['outlier_fraction']):.0f}"}
    (PAPER/'numbers.tex').write_text('% Generated by bin/build_apjl_assets.py; do not edit numerical values here.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in values.items()))
    print('Frozen figure inputs and generated table/macros; test IDs and predictions verified.')

if __name__=='__main__':main()
