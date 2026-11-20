"""Check grid reconstruction sensitivity and TARP on self-consistent PDF draws."""
import json
import numpy as np
from .cosmos2020_coverage import DATA,OUT,SEED,ALPHA,GridPDF,tarp_masses,contributions

c=np.load(DATA/'classic_matched_pdfs.npz')
a=np.load(OUT/'coverage_arrays.npz')
assert np.array_equal(c['ids'],a['id'])
g=GridPDF(c['grid'],c['pdf'])
z=a['zspec']
# Alternate representation: constant densities in bins centered on grid points.
edges=np.r_[0,(c['grid'][1:]+c['grid'][:-1])/2,10]
p=c['pdf'].astype(float)
mass=p*np.diff(edges)
cdf=np.c_[np.zeros(len(p)),np.cumsum(mass,axis=1)]
cdf/=cdf[:,-1:]
ii=np.clip(np.searchsorted(edges,z,side='right')-1,0,len(edges)-2)
row=np.arange(len(z))
pit=cdf[row,ii]+(cdf[row,ii+1]-cdf[row,ii])*(z-edges[ii])/np.diff(edges)[ii]
groups=[('all',np.ones(len(z),bool)),('i<22.5',(a['mag_i']>0)&(a['mag_i']<22.5)),('i>=25',(a['mag_i']>=25)&(a['mag_i']<40))]
result={'alternate_grid_model':'piecewise-constant density on midpoint-centered grid cells',
        'coverage_sensitivity':{name:{str(alpha):float(np.mean(abs(2*pit[m]-1)<=alpha)) for alpha in [.68,.95]} for name,m in groups}}
# Independent truths generated from the actual PDFs must produce calibrated curves.
rng=np.random.default_rng(SEED+998)
zsim=g.quantile(rng.random(len(z)))
center=g.quantile(.5)
m=tarp_masses(g,zsim,center,16,SEED+999)
result['actual_pdf_self_consistency_max_deviation']={name:float(np.max(abs(contributions(m[k]).mean(axis=0)-ALPHA))) for k,name in enumerate(['global','local'])}
result['actual_pdf_self_consistency_central68']=float(np.mean(abs(2*g.cdf(zsim)-1)<=.68))
assert max(result['actual_pdf_self_consistency_max_deviation'].values())<.02
assert abs(result['actual_pdf_self_consistency_central68']-.68)<.02
(OUT/'numerical_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
