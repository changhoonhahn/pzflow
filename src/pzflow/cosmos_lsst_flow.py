"""Conditional normalizing-flow transfer of COSMOS redshift PDFs to six bands.

Run flow_experiment_data.py first. This script supports either actual matched
Rubin fluxes or a prominently labeled native-filter noise forecast.
"""
from pathlib import Path
import argparse
import copy
import json
import time
import numpy as np
import torch
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/flow_experiment'
SEED=20261024

def make_flow(context_features=12):
    transforms=[MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
        features=1,hidden_features=64,context_features=context_features,num_bins=16,
        tails='linear',tail_bound=5.,num_blocks=2,dropout_probability=.0,
        use_batch_norm=False) for _ in range(2)]
    return Flow(CompositeTransform(transforms),StandardNormal([1]))

def z_to_t(z):
    z=torch.clamp(z,1e-5,10-1e-5)
    return (torch.log(z/(10-z))+2.2)/1.1

def t_to_z(t):return 10*torch.sigmoid(t*1.1-2.2)

class PDFSampler:
    def __init__(self,grid,pdf):
        self.grid=grid
        area=(pdf[:,:-1]+pdf[:,1:])*np.diff(grid)[None,:]/2
        self.c=np.c_[np.zeros(len(pdf)),np.cumsum(area,axis=1)].astype(np.float32)
        norm=self.c[:,-1].copy();assert np.all(norm>0)
        self.c/=norm[:,None]
        self.p=(pdf/norm[:,None]).astype(np.float32)
    def draw(self,idx,rng):
        u=rng.random(len(idx));lo=np.zeros(len(idx),int);hi=np.full(len(idx),len(self.grid)-1,int)
        for _ in range(11):
            mid=(lo+hi)//2;right=self.c[idx,mid]<=u
            lo=np.where(right,mid,lo);hi=np.where(right,hi,mid)
        j=np.minimum(lo,len(self.grid)-2)
        a=self.p[idx,j];s=(self.p[idx,j+1]-a)/np.diff(self.grid)[j]
        w=np.maximum(u-self.c[idx,j],0)
        den=a+np.sqrt(np.maximum(a*a+2*s*w,0))
        t=np.divide(2*w,den,out=np.zeros(len(idx)),where=den>0)
        return (self.grid[j]+np.clip(t,0,np.diff(self.grid)[j])).astype(np.float32)

def raw_features(flux,err):
    soft=.02 # microJy; fixed and independent of labels
    lupt=23.9-(2.5/np.log(10))*(np.arcsinh(flux/(2*soft))+np.log(soft))
    return np.c_[lupt,np.log(np.maximum(err,1e-7)/soft)].astype(np.float32)

def context(flux,err,mean,scale):
    return torch.from_numpy(np.clip((raw_features(flux,err)-mean)/scale,-12,12).astype(np.float32))

@torch.no_grad()
def log_prob_z(model,z,ctx):
    z=torch.as_tensor(z,dtype=torch.float32).reshape(-1,1)
    w=z_to_t(z)
    jac=np.log(10)-torch.log(z)-torch.log(10-z)-np.log(1.1)
    return model.log_prob(w,context=ctx)+jac[:,0]

@torch.no_grad()
def posterior_summary(model,ctx,zspec):
    probs=torch.tensor([.025,.16,.5,.84,.975])
    noise=torch.distributions.Normal(0.,1.).icdf(probs)
    quant=[];pit=[];nll=[]
    for start in range(0,len(ctx),512):
        x=ctx[start:start+512];n=len(x)
        q,_=model._transform.inverse(noise.repeat(n).reshape(-1,1),context=x.repeat_interleave(5,dim=0))
        quant.append(t_to_z(q).reshape(n,5).numpy())
        z=torch.tensor(zspec[start:start+512],dtype=torch.float32).reshape(-1,1)
        u,_=model._transform(z_to_t(z),context=x)
        pit.append((.5*(1+torch.erf(u[:,0]/np.sqrt(2)))).numpy())
        nll.append((-log_prob_z(model,z,x)).numpy())
    return np.concatenate(quant),np.concatenate(pit),np.concatenate(nll)

def fit(name,teacher,flux,err,extra,zspec,splits,sampler,mean,scale,epochs,max_steps,matched=False):
    rng=np.random.default_rng(SEED)
    train=np.flatnonzero(splits==0);valid=np.flatnonzero(splits==1)
    if not teacher or matched:
        train=train[np.isfinite(zspec[train])];valid=valid[np.isfinite(zspec[valid])]
    assert len(train)>100 and len(valid)>100
    torch.manual_seed(SEED)
    model=make_flow();opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=3,factor=.5)
    # Fixed validation labels and noise; no spec-z used for teacher validation.
    vi=rng.choice(valid,min(len(valid),4096),replace=False)
    if teacher:vi=np.repeat(vi,4)
    vy=sampler.draw(vi,rng) if teacher else zspec[vi].astype(np.float32)
    vf=flux[vi]+rng.normal(size=(len(vi),6)).astype(np.float32)*extra[vi]
    vx=context(vf,err[vi],mean,scale);vt=z_to_t(torch.from_numpy(vy[:,None]))
    best=float('inf');best_state=None;bad=0;step=0;history=[];start=time.monotonic()
    for epoch in range(epochs):
        model.train();rng.shuffle(train);losses=[]
        for pos in range(0,len(train),512):
            idx=train[pos:pos+512]
            xflux=flux[idx]+rng.normal(size=(len(idx),6)).astype(np.float32)*extra[idx]
            x=context(xflux,err[idx],mean,scale)
            z=sampler.draw(idx,rng) if teacher else zspec[idx].astype(np.float32)
            target=z_to_t(torch.from_numpy(z[:,None]))
            opt.zero_grad(set_to_none=True)
            loss=-model.log_prob(target,context=x).mean()
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss in {name}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step();losses.append(float(loss.detach()));step+=1
            if step>=max_steps:break
        model.eval()
        with torch.no_grad():
            vals=[-model.log_prob(vt[p:p+1024],context=vx[p:p+1024]).sum().item() for p in range(0,len(vt),1024)]
            val=sum(vals)/len(vt)
        scheduler.step(val)
        history.append(dict(epoch=epoch+1,step=step,train_loss=float(np.mean(losses)),validation_loss=val))
        print(f'{name} epoch={epoch+1} step={step} train={np.mean(losses):.4f} val={val:.4f}',flush=True)
        if val<best-1e-4:best=val;best_state=copy.deepcopy(model.state_dict());bad=0
        else:bad+=1
        if bad>=8 or step>=max_steps:break
    model.load_state_dict(best_state);model.eval()
    torch.save(dict(state_dict=best_state,feature_mean=mean,feature_scale=scale,
                    model='two conditional RQS autoregressive transforms; 16 bins; hidden64; blocks2',
                    n_train=len(train),n_validation=len(valid),seed=SEED),OUT/f'{name}.pt')
    (OUT/f'{name}_history.json').write_text(json.dumps(history,indent=2)+'\n')
    return model,dict(n_train=len(train),n_validation=len(valid),steps=step,epochs=len(history),best_validation_loss=best,seconds=time.monotonic()-start)

def metrics(z,q,pit,nll):
    r=(q[:,2]-z)/(1+z)
    return dict(n=len(z),bias=float(np.median(r)),sigma_nmad=float(1.4826*np.median(abs(r-np.median(r)))),
                outlier_fraction=float(np.mean(abs(r)>.15)),rmse_normalized=float(np.sqrt(np.mean(r*r))),
                coverage68=float(np.mean((z>=q[:,1])&(z<=q[:,3]))),coverage95=float(np.mean((z>=q[:,0])&(z<=q[:,4]))),
                mean_negative_log_density=float(np.mean(nll)),median_width68=float(np.median(q[:,3]-q[:,1])))

def main(epochs=200,max_steps=20000,data_dir=None,output_dir=None):
    global OUT
    data_dir=data_dir or ROOT/'data/flow_magnitude_limited'
    OUT=output_dir or ROOT/'results/flow_magnitude_limited'
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(SEED)
    d=np.load(data_dir/'prepared.npz')
    ids=d['id'];flux=d['flux'];err=d['err'];extra=d['extra'];zspec=d['zspec'];splits=d['split'];cell=d['cell']
    for a in range(3):
        for b in range(a):assert not np.intersect1d(cell[splits==a],cell[splits==b]).size
    base=raw_features(flux[splits==0],err[splits==0]);mean=base.mean(axis=0);scale=np.maximum(base.std(axis=0),.1)
    sampler=PDFSampler(d['grid'],d['pdf'])
    test=np.flatnonzero((splits==2)&np.isfinite(zspec))
    assert len(test)>100
    rng=np.random.default_rng(SEED+1)
    observed=flux[test]+rng.normal(size=(len(test),6)).astype(np.float32)*extra[test]
    ctx=context(observed,err[test],mean,scale)
    outputs={};report={}
    for name,teacher,matched in [('cosmos_pdf_flow',True,False),('cosmos_pdf_matched_flow',True,True),('broadband_spec_flow',False,True)]:
        model,training=fit(name,teacher,flux,err,extra,zspec,splits,sampler,mean,scale,epochs,max_steps,matched)
        q,pit,nll=posterior_summary(model,ctx,zspec[test])
        outputs[name]=dict(quantiles=q,pit=pit,nll=nll)
        report[name]=dict(training=training,all=metrics(zspec[test],q,pit,nll),
                          z_gt2=metrics(zspec[test][zspec[test]>2],q[zspec[test]>2],pit[zspec[test]>2],nll[zspec[test]>2]))
        print(name,json.dumps(report[name]),flush=True)
    # Paired resampling of common test galaxies; no metric is selected on test.
    paired={};rng=np.random.default_rng(SEED+2)
    for teacher_name in ['cosmos_pdf_flow','cosmos_pdf_matched_flow']:
        paired[teacher_name]={}
        for key in ['sigma_nmad','outlier_fraction','mean_negative_log_density']:
            differences=[]
            for _ in range(200):
                ix=rng.integers(0,len(test),len(test));vv=[]
                for name in [teacher_name,'broadband_spec_flow']:
                    o=outputs[name];vv.append(metrics(zspec[test][ix],o['quantiles'][ix],o['pit'][ix],o['nll'][ix])[key])
                differences.append(vv[0]-vv[1])
            paired[teacher_name][key]=dict(flow_minus_baseline_95ci=np.quantile(differences,[.025,.975]).tolist())
    manifest=json.loads((data_dir/'manifest.json').read_text())
    result=dict(data=manifest,models=report,paired_bootstrap=paired,
                run_settings=dict(seed=SEED,max_epochs=epochs,max_steps=max_steps,n_training_seeds=1),
                limitations=['Forecast uses native COSMOS filters, not measured LSST bandpasses' if manifest['mode']!='actual_rubin' else 'Matched Rubin fluxes supplied by user',
                             'Spectroscopic test selection is not representative of the full photometric population',
                             'Held-out IDs are excluded from our model fitting; independence from the original COSMOS template/zero-point calibration has not been audited',
                             'COSMOS redshift distributions are assumed unbiased; their supplied widths and tails are retained without recalibration',
                             'Baseline is a spec-supervised six-band flow, not an official Rubin photo-z product or a template-fitting code'])
    (OUT/'comparison_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    arrays=dict(id=ids[test],zspec=zspec[test],imag=d['imag'][test],observed_flux=observed,fluxerr=err[test])
    for name,o in outputs.items():
        for k,v in o.items():arrays[f'{name}_{k}']=v
    np.savez_compressed(OUT/'heldout_predictions.npz',**arrays)
    np.savez_compressed(OUT/'split_ids.npz',**{f'split_{s}':ids[splits==s] for s in range(3)})
    np.savez_compressed(OUT/'matched_training_ids.npz',**{f'split_{s}':ids[(splits==s)&np.isfinite(zspec)] for s in [0,1]})
    make_plots(arrays,result)

def make_plots(a,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .plot_style import set_style, COLORS
    set_style()
    forecast=result['data']['mode']!='actual_rubin'
    title='LSST-like noise forecast · native COSMOS filters (not Rubin photometry)' if forecast else 'Matched Rubin photometry experiment'
    names=[n for n in ['broadband_spec_flow','cosmos_pdf_flow','cosmos_pdf_matched_flow'] if f'{n}_pit' in a]
    current='cosmos_pdf_matched_flow' in names
    labels=['Spec-supervised flow','COSMOS PDFs, full magnitude-limited sample' if current else 'COSMOS PDFs, archived training sample','COSMOS PDFs, same objects as spec baseline'][:len(names)]
    fig,axs=plt.subplots(1,3,figsize=(15,5.3),layout='constrained')
    z=a['zspec']
    for ax,name,label in zip(axs[:2],names,labels):
        pred=a[f'{name}_quantiles'][:,2]
        h=ax.hexbin(z,pred,gridsize=100,bins='log',mincnt=1,cmap='viridis',linewidths=0,rasterized=True)
        ax.plot([0,8],[0,8],color='C0',lw=1)
        ax.set(xlim=(0,5),ylim=(0,5),xlabel='Held-out spectroscopic redshift',ylabel='Predicted median redshift',title=label)
        m=result['models'][name]['all']
        ax.text(.04,.96,f"N={len(z):,}\nNMAD={m['sigma_nmad']:.4f}\nOutliers={100*m['outlier_fraction']:.1f}%",transform=ax.transAxes,va='top',bbox=dict(facecolor='white',alpha=.9,edgecolor='0.8'))
    alpha=np.linspace(0,1,101)
    for name,label in zip(names,labels):
        color=COLORS[name]
        rank=abs(2*a[f'{name}_pit']-1)
        axs[2].plot(alpha,[(rank<=v).mean() for v in alpha],color=color,label=label)
    axs[2].plot([0,1],[0,1],'k--',lw=1)
    axs[2].set(xlabel='Nominal credibility',ylabel='Empirical equal-tailed coverage',title='Common held-out sample',xlim=(0,1),ylim=(0,1))
    axs[2].legend(fontsize=8);fig.suptitle(title,fontsize=15,fontweight='bold')
    note='Clean galaxy catalog, HSC i (2 arcsec aperture) < 25 AB, i S/N >= 5. No spectroscopic oversampling in full COSMOS training.' if current else 'Archived experiment: earlier spectroscopically enriched training selection; superseded by the magnitude-limited census.'
    fig.supxlabel(note+'\nDisjoint sky cells; only the secure-spec test subset is scored. Scatter panels zoom to z<5; metrics use all test galaxies. COSMOS redshifts assumed unbiased.',fontsize=9)
    fig.savefig(OUT/'flow_vs_broadband.png',dpi=180);fig.savefig(OUT/'flow_vs_broadband.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for name,label in zip(names,labels):
        color=COLORS[name]
        pred=a[f'{name}_quantiles'][:,2];r=(pred-z)/(1+z)
        axs[0].hist(r,bins=np.linspace(-.3,.3,81),density=True,histtype='step',color=color,label=label,lw=1.7)
        axs[1].hist(a[f'{name}_pit'],bins=20,range=(0,1),density=True,histtype='step',color=color,label=label,lw=1.7)
    axs[0].set(xlabel=r'$(z_{\rm pred}-z_{\rm spec})/(1+z_{\rm spec})$',ylabel='Density',title='Median-redshift residuals')
    axs[0].legend(fontsize=9);axs[1].axhline(1,color='k',ls='--',lw=1)
    axs[1].set(xlabel='PIT',ylabel='Density',title='Predictive PDF calibration')
    fig.suptitle(title,fontsize=13);fig.savefig(OUT/'residuals_and_pit.png',dpi=180);fig.savefig(OUT/'residuals_and_pit.pdf');plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=200);parser.add_argument('--max-steps',type=int,default=20000)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/flow_magnitude_limited')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/flow_magnitude_limited')
    args=parser.parse_args();main(args.epochs,args.max_steps,args.data_dir,args.output_dir)
