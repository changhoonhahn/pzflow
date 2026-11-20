"""Meaningful numerical checks for the PDF sampler and conditional flow."""
import unittest
import numpy as np
import torch
from .cosmos_lsst_flow import make_flow,z_to_t,t_to_z,PDFSampler
from .flow_experiment_data import sample_pdf,magnitude_parent

class FlowExperimentTests(unittest.TestCase):
    def test_magnitude_selection(self):
        bands=['cfht_u','hsc_g','hsc_r','hsc_i','hsc_z','hsc_y']
        dtype=[('flag_combined',int),('lp_type',int),('hsc_i_flux_aper2',float)]+[(f'{b}_fluxerr_aper2',float) for b in bands]
        raw=np.zeros(8,dtype=dtype);raw['hsc_i_flux_aper2']=1.
        for b in bands:raw[f'{b}_fluxerr_aper2']=.01
        flux=np.ones((8,6));err=np.full((8,6),.01)
        flux[0,0]=-1 # a u-band nondetection is allowed
        raw['hsc_i_flux_aper2'][1]=.1 # too faint
        raw['flag_combined'][2]=1;raw['lp_type'][3]=1
        raw['hsc_i_fluxerr_aper2'][4]=1 # insufficient i SNR
        flux[5,0]=np.nan;raw['hsc_y_fluxerr_aper2'][6]=-1
        keep,_=magnitude_parent(raw,flux,err)
        np.testing.assert_array_equal(keep,[True,False,False,False,False,False,False,True])
        perm=np.random.default_rng(8).permutation(len(raw))
        perm_keep,_=magnitude_parent(raw[perm],flux[perm],err[perm])
        np.testing.assert_array_equal(perm_keep,keep[perm])
    def test_pdf_sampling_mixture(self):
        grid=np.linspace(0,10,1001)
        p=.8*np.exp(-.5*((grid-1)/.08)**2)/.08+.2*np.exp(-.5*((grid-3)/.15)**2)/.15
        sampler=PDFSampler(grid,p[None,:])
        draw=sampler.draw(np.zeros(20000,int),np.random.default_rng(5))
        self.assertAlmostEqual(float(np.mean(draw<2)),.8,delta=.015)
        draw2=sample_pdf(grid,np.tile(p,(10000,1)),np.random.default_rng(6))
        self.assertAlmostEqual(float(np.mean(draw2<2)),.8,delta=.02)
    def test_flow_inverse_and_normalization(self):
        torch.set_num_threads(4);torch.manual_seed(7)
        flow=make_flow();flow.eval()
        x=torch.randn(80,12);z=torch.linspace(.02,9.98,80)[:,None]
        w=z_to_t(z);noise,_=flow._transform(w,context=x)
        recovered,_=flow._transform.inverse(noise,context=x)
        self.assertLess(float(abs(t_to_z(recovered)-z).max()),1e-4)
        u1,_=flow._transform(z_to_t(torch.full((80,1),1e-5)),context=x)
        u2,_=flow._transform(z_to_t(torch.full((80,1),9.99999)),context=x)
        cdf=lambda a:.5*(1+torch.erf(a/np.sqrt(2)))
        self.assertTrue(torch.all(cdf(u1)<1e-5))
        self.assertTrue(torch.all(cdf(u2)>1-1e-5))
    def test_conditional_learning(self):
        torch.set_num_threads(4);torch.manual_seed(8)
        flow=make_flow();opt=torch.optim.Adam(flow.parameters(),lr=1e-3)
        vx=torch.randn(1024,12);vy=.8*vx[:,0:1]+.15*torch.randn(1024,1)
        with torch.no_grad():before=-flow.log_prob(vy,context=vx).mean().item()
        for _ in range(180):
            x=torch.randn(256,12);y=.8*x[:,0:1]+.15*torch.randn(256,1)
            loss=-flow.log_prob(y,context=x).mean()
            opt.zero_grad();loss.backward();opt.step()
        flow.eval()
        with torch.no_grad():
            after=-flow.log_prob(vy,context=vx).mean().item()
            median,_=flow._transform.inverse(torch.zeros(1024,1),context=vx)
        self.assertLess(after,before-.5)
        self.assertGreater(np.corrcoef(median[:,0].numpy(),vx[:,0].numpy())[0,1],.95)

if __name__=='__main__':unittest.main()
