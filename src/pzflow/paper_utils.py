"""Small shared data/style helpers; plotting code lives in the individual notebooks."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from .project_plot_style import COLORS, set_style

LABELS={'cosmos_pdf_flow':'Bootstrapped flow','cosmos_pdf_matched_flow':'COSMOS PDFs: matched',
        'broadband_spec_flow':'Broadband flow','lephare_full_cosmos':'Full-COSMOS LePhare',
        'eazy_rprior':'EAZY: R prior','eazy_six_band':'EAZY: flat prior'}

def load_predictions(root):
    p=Path(root)/'paper/data/predictions.npz'
    with np.load(p) as a:result={k:a[k] for k in a.files}
    assert len(np.unique(result['id']))==len(result['id'])==5083
    for k,v in result.items():
        if k.endswith('_quantiles'):assert np.isfinite(v).all() and np.all(np.diff(v,axis=1)>=0)
    return result

def load_metrics(root):return json.loads((Path(root)/'paper/data/metrics.json').read_text())

def summary(z,q):
    r=(q[:,2]-z)/(1+z)
    return {'n':len(z),'bias':float(np.median(r)),
            'sigma_nmad':float(1.4826*np.median(abs(r-np.median(r)))),
            'outlier_fraction':float(np.mean(abs(r)>.15)),
            'coverage68':float(np.mean((z>=q[:,1])&(z<=q[:,3]))),
            'coverage95':float(np.mean((z>=q[:,0])&(z<=q[:,4])))}

def bootstrap(z,q,seed=20261117,n_boot=200):
    rng=np.random.default_rng(seed);keys=['sigma_nmad','coverage68'];values=[]
    for _ in range(n_boot):
        ix=rng.integers(0,len(z),len(z));m=summary(z[ix],q[ix]);values.append([m[k] for k in keys])
    return dict(zip(keys,np.quantile(values,[.025,.975],axis=0).T))

def save_figure(fig,root,stem):
    out=Path(root)/'paper/figures';out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(out/f'{stem}.{ext}',bbox_inches='tight',pad_inches=.025)
    print(f'Saved paper/figures/{stem}.pdf and .png')
