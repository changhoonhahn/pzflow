"""Shared SpenderQ methods-paper plotting style for every project figure.

Reference: galactic-ai/SpenderQ, doc/methods_paper/{validation,uncertainty}.ipynb
Retrieved 2026-11-19. The reference uses LaTeX serif text, 1.5-pt axes/ticks,
5-pt major ticks, frameless legends, and the standard Matplotlib color cycle.
Figure-specific axis ranges, statistics, data, and uncertainty estimates remain
the responsibility of the figure code. Requires latex and dvipng on PATH.
"""
import re
import shutil
import os
from pathlib import Path
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text

COLORS={'cosmos_pdf_flow':'C0','cosmos_pdf_matched_flow':'C9',
        'broadband_spec_flow':'C1','lephare_full_cosmos':'C3',
        'eazy_rprior':'C2','eazy_six_band':'C4'}


def tex_text(s):
    """Make plain annotations safe for the same LaTeX renderer as the reference."""
    replacements={'→':r'\ensuremath{\rightarrow}', '≥':r'\ensuremath{\geq}',
                  '≤':r'\ensuremath{\leq}', 'σ':r'\ensuremath{\sigma}',
                  '±':r'\ensuremath{\pm}', '×':r'\ensuremath{\times}',
                  '∼':r'\ensuremath{\sim}', '″':"''", '′':"'",
                  '–':'--','—':'---','−':'-', '·':r'\ensuremath{\cdot}',
                  '\u2009':r'\,', '\u00a0':' '}
    for char,value in replacements.items():s=s.replace(char,value)
    s=re.sub(r'(?<!\\)%',r'\\%',s)
    parts=re.split(r'(?<!\\)(\$)',s)
    in_math=False
    for i,part in enumerate(parts):
        if part=='$':in_math=not in_math
        elif not in_math:
            parts[i]=re.sub(r'(?<!\\)_',r'\\_',part)
            parts[i]=re.sub(r'(?<!\\)&',r'\\&',parts[i])
            parts[i]=re.sub(r'(?<!\\ensuremath\{)([<>])',
                            lambda m:r'\ensuremath{'+m.group(1)+'}',parts[i])
    return ''.join(parts)


def finish_figure(fig):
    """Apply consistent text sizes after figure-local formatting, before export."""
    # The old paper figures were authored at half the reference's display size.
    if fig.get_figwidth()<8 and not getattr(fig,'_project_size_scaled',False):
        fig.set_size_inches(fig.get_size_inches()*2)
        fig._project_size_scaled=True
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(True);spine.set_linewidth(1.5)
        ax.tick_params(axis='both',which='major',labelsize=13,length=5,width=1.5)
        ax.tick_params(axis='both',which='minor',length=3,width=1.)
        ax.xaxis.label.set_fontsize(17);ax.yaxis.label.set_fontsize(17)
        ax.title.set_fontsize(15);ax.title.set_fontweight('normal')
        ax.grid(False)
        legend=ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            for text in legend.get_texts():text.set_fontsize(11)
        for text in ax.texts:text.set_fontsize(11)
    for attr,size in [('_suptitle',17),('_supxlabel',10),('_supylabel',17)]:
        text=getattr(fig,attr,None)
        if text is not None:text.set_fontsize(size);text.set_fontweight('normal')
    for text in fig.findobj(match=Text):
        text.set_text(tex_text(text.get_text()))
        text.set_fontfamily('serif')
    return fig


def set_style():
    if not shutil.which('latex') and Path('/Library/TeX/texbin/latex').exists():
        os.environ['PATH']='/Library/TeX/texbin'+os.pathsep+os.environ.get('PATH','')
    if not shutil.which('latex') or not shutil.which('dvipng'):
        raise RuntimeError('This plotting style requires a TeX installation with latex and dvipng on PATH.')
    mpl.rcParams.update({'text.usetex':True,'font.family':'serif',
        'font.size':12,'axes.linewidth':1.5,'axes.labelsize':17,'axes.titlesize':15,
        'axes.spines.top':True,'axes.spines.right':True,'axes.grid':False,
        'xtick.labelsize':'x-large','ytick.labelsize':'x-large',
        'xtick.major.size':5,'ytick.major.size':5,
        'xtick.major.width':1.5,'ytick.major.width':1.5,
        'legend.frameon':False,'legend.fontsize':11,'lines.linewidth':1.5,
        'savefig.dpi':220,'pdf.fonttype':42,'ps.fonttype':42,
        'text.latex.preamble':r'\usepackage{amsmath}',
        'axes.prop_cycle':mpl.cycler(color=mpl.colormaps['tab10'].colors)})
    if not getattr(Figure.savefig,'_project_style',False):
        original=Figure.savefig
        def savefig(self,*args,**kwargs):
            finish_figure(self)
            kwargs.setdefault('bbox_inches','tight')
            return original(self,*args,**kwargs)
        savefig._project_style=True
        Figure.savefig=savefig


import json
import numpy as np
import matplotlib.pyplot as plt

LABELS={'cosmos_pdf_flow': 'Bootstrapped flow', 'cosmos_pdf_matched_flow': 'COSMOS PDFs: matched', 'broadband_spec_flow': 'Broadband flow', 'lephare_full_cosmos': 'Full-COSMOS LePhare', 'eazy_rprior': 'EAZY: R prior', 'eazy_six_band': 'EAZY: flat prior'}
def load_predictions(root):
    p=Path(root)/'results/figure_data/predictions.npz'
    with np.load(p) as a:result={k:a[k] for k in a.files}
    assert len(np.unique(result['id']))==len(result['id'])==5083
    for k,v in result.items():
        if k.endswith('_quantiles'):assert np.isfinite(v).all() and np.all(np.diff(v,axis=1)>=0)
    return result


def load_metrics(root):return json.loads((Path(root)/'results/figure_data/metrics.json').read_text())


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
    out=Path(root)/'tex/figures';out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(out/f'{stem}.{ext}',bbox_inches='tight',pad_inches=.025)
    print(f'Saved tex/figures/{stem}.pdf and .png')
