"""Redshift-prior sensitivity utilities shared by the analysis scripts."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .paths import ROOT
from .pdf import GridPDF
SEED=20261118
WIDTHS=[.015,.03,.06,.10]
EPS=.001

def smooth_density(t, p, width):
    p = gaussian_filter1d(p, width / (t[1] - t[0]), mode='reflect')
    p /= np.trapezoid(p, t)
    return (1-EPS)*p + EPS/(t[-1]-t[0])


def get_summary(z, grid, p):
    g = GridPDF(grid, p.copy())
    pit = g.cdf(z)
    med = g.quantile(.5)
    r = (med-z)/(1+z)
    rank = np.abs(2*pit-1)
    row = dict(n=len(z), coverage68=float(np.mean(rank <= .68)),
               coverage95=float(np.mean(rank <= .95)),
               bias=float(np.median(r)),
               sigma_nmad=float(1.4826*np.median(abs(r-np.median(r)))),
               pit_mean=float(pit.mean()),
               extreme_pit_fraction=float(np.mean((pit < .025)|(pit > .975))))
    return row, pit


def reweight_flow(u,zq,pit,ratio_grid,ratio):
    w=np.interp(zq,ratio_grid,ratio)
    du=np.diff(u)
    c=np.c_[np.zeros(len(zq)),np.cumsum(.5*(w[:,:-1]+w[:,1:])*du,axis=1)]
    j=np.clip(np.searchsorted(u,pit,side='right')-1,0,len(u)-2)
    row=np.arange(len(pit));delta=pit-u[j]
    val=c[row,j]+w[row,j]*delta+.5*(w[row,j+1]-w[row,j])*delta**2/du[j]
    return np.clip(val/c[:,-1],0,1)


def pit_summary(pit):
    return dict(n=len(pit),coverage68=float(np.mean(abs(2*pit-1)<=.68)),
                coverage95=float(np.mean(abs(2*pit-1)<=.95)),pit_mean=float(pit.mean()))
