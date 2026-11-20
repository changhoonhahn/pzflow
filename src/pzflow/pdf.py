"""Gridded PDFs, coverage/PIT, TARP, and numerical controls."""
import numpy as np
SEED=20261023
ALPHA=np.unique(np.r_[np.linspace(0,1,101),.68,.95])

class GridPDF:
    def __init__(self,grid,pdf):
        self.grid=np.asarray(grid,float)
        self.p=np.asarray(pdf,float)
        masses=np.diff(self.grid)[None,:]*(self.p[:,:-1]+self.p[:,1:])/2
        self.c=np.c_[np.zeros(len(pdf)),np.cumsum(masses,axis=1)]
        self.norm=self.c[:,-1].copy()
        assert np.all(np.isfinite(self.p)) and np.all(self.p>=0)
        assert np.all(self.norm>0)
        self.p/=self.norm[:,None];self.c/=self.norm[:,None]
    def cdf(self,x):
        x=np.asarray(x,float)
        assert x.shape==(len(self.p),)
        xclip=np.clip(x,self.grid[0],self.grid[-1])
        j=np.clip(np.searchsorted(self.grid,xclip,side='right')-1,0,len(self.grid)-2)
        row=np.arange(len(self.p));t=xclip-self.grid[j]
        a=self.p[row,j]; b=self.p[row,j+1]
        result=self.c[row,j]+a*t+.5*(b-a)/np.diff(self.grid)[j]*t*t
        return np.clip(result,0,1)
    def quantile(self,u):
        u=np.broadcast_to(u,(len(self.p),))
        lo=np.zeros(len(self.p));hi=np.full(len(self.p),self.grid[-1])
        for _ in range(40):
            mid=(lo+hi)/2
            lower=self.cdf(mid)<u
            lo=np.where(lower,mid,lo);hi=np.where(lower,hi,mid)
        return (lo+hi)/2


def tarp_masses(g,z,center,refs,seed):
    rng=np.random.default_rng(seed)
    out=[]
    for _ in range(refs):
        global_ref=rng.uniform(0,10,len(z))
        local_ref=rng.uniform(np.maximum(0,center-.25*(1+center)),np.minimum(10,center+.25*(1+center)))
        both=[]
        for ref in [global_ref,local_ref]:
            radius=np.abs(z-ref)
            both.append(np.clip(g.cdf(ref+radius)-g.cdf(ref-radius),0,1))
        out.append(both)
    return np.asarray(out).transpose(1,2,0) # scheme, galaxy, random-reference


def contributions(mass):
    if mass.ndim==1:mass=mass[:,None]
    a=np.empty((len(mass),len(ALPHA)),np.float32)
    for k,alpha in enumerate(ALPHA):a[:,k]=(mass<=alpha).mean(axis=1)
    # Set endpoints for display; finite-precision exact zero/one tails are audited separately.
    a[:,0]=0;a[:,-1]=1
    return a


def bootstrap(curves,mask,seed):
    x=curves[mask]
    rng=np.random.default_rng(seed)
    # Resample galaxies, retaining all reference draws for each selected galaxy.
    draws=rng.multinomial(len(x),np.full(len(x),1/len(x)),size=200).astype(np.float32)
    boots=draws@x/len(x)
    return x.mean(axis=0),np.quantile(boots,[.025,.975],axis=0)


def controls():
    rng=np.random.default_rng(SEED)
    n=5000;grid=np.linspace(0,10,1001)
    mu=rng.uniform(.5,3,n)
    sigma=rng.uniform(.04,.3,n)
    pdf=np.exp(-.5*((grid[None,:]-mu[:,None])/sigma[:,None])**2)
    g=GridPDF(grid,pdf)
    u=rng.random(n)
    z=g.quantile(u)
    pit=g.cdf(z)
    assert np.max(np.abs(pit-u))<1e-8,'CDF/quantile inversion failed'
    m=tarp_masses(g,z,mu,8,SEED+1)
    curves=np.array([contributions(x).mean(axis=0) for x in m])
    assert np.max(abs(curves-ALPHA))<.035,'Calibrated synthetic control failed'
    narrow=GridPDF(grid,pdf**4)
    nc=contributions(np.abs(2*narrow.cdf(z)-1)).mean(axis=0)
    assert nc[np.argmin(abs(ALPHA-.68))]<.5,'Overconfidence control failed'
    return dict(calibrated_tarp_max_deviation=float(np.max(abs(curves-ALPHA))),
                narrowed_equal_tailed_68=float(nc[np.argmin(abs(ALPHA-.68))]))
