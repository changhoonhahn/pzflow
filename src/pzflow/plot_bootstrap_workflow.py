"""Draw the manuscript's vector workflow diagram; no model fitting is required."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]

def main():
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    fig.subplots_adjust(left=.01, right=.99, bottom=.025, top=.99)
    ax.set(xlim=(0, 10), ylim=(1.2, 3.85))
    ax.axis('off')
    def box(x, y, w, h, text, color='#edf3f8', edge='#46627a'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.07',
                                   facecolor=color,edgecolor=edge,linewidth=.85))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',linespacing=1.4,fontsize=8)
    def arrow(a,b, dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,
                                    linewidth=.85,color='#425466',linestyle='--' if dashed else '-'))
    ax.text(.1,3.78,'TRAINING',fontsize=8.5,weight='bold',color='#33475b')
    box(.15,2.15,1.7,.8,'Deep COSMOS\nobservations '+r'$y_i$')
    box(2.4,2.95,2.35,.64,'Draw redshift label\n'+r'$z_i^{\prime}\sim p_{\rm C}(z\mid y_i)$')
    box(2.4,1.65,2.35,.78,'Broadband photometry\n'+r'$x_i^{\prime}$')
    arrow((1.85,2.75),(2.35,3.23))
    arrow((1.85,2.35),(2.35,2.05))
    box(5.35,2.17,1.35,.76,'Training pair\n'+r'$(x_i^{\prime},z_i^{\prime})$')
    arrow((4.8,3.22),(5.3,2.77))
    arrow((4.8,2.05),(5.3,2.4))
    box(7.35,2.04,2.35,1.03,'Fit conditional flow\n'+r'$q_\theta(z\mid x)$'+'\nMaximize '+r'$\log q_\theta(z_i^{\prime}\mid x_i^{\prime})$',
        color='#fcf0df',edge='#9c6b32')
    arrow((6.75,2.55),(7.3,2.55))
    ax.text(3.57,1.39,'New redshift and noise draws each epoch',ha='center',fontsize=7.4,color='#425466')
    for ext in ['pdf','png']:
        out=ROOT/'paper/figures'/f'fig05_bootstrap_workflow.{ext}'
        fig.savefig(out,dpi=220,bbox_inches='tight',pad_inches=.03)
        print(out)
    plt.close(fig)

if __name__=='__main__':main()
