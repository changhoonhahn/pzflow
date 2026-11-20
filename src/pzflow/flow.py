"""Conditional flow, PDF-label sampling, training, and predictive diagnostics."""
from pathlib import Path
import copy, json, time
import numpy as np
import torch
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from .paths import ROOT
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
