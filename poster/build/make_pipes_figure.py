"""Clearer 'open space vs electrons' figure: same channel twice, electrons connected or in pockets."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import sys
from pathlib import Path
OUT=sys.argv[1] if len(sys.argv)>1 else str(Path(__file__).resolve().parents[1]/'figures'/'pipes_vs_electrons')
BG,OPEN,ATOM,EDGE,GOLD,INK,MUTED='#f2f2f2','#e0e0e0','#9d9795','#555960','#daaa00','#000000','#555960'
plt.rcParams.update({'font.family':'Arial','text.color':INK})
fig=plt.figure(figsize=(12,7.0),facecolor=BG);ax=fig.add_axes((0,0,1,1));ax.set_facecolor(BG)
ax.set_xlim(-0.7,12.6);ax.set_ylim(-4.75,3.0);ax.set_aspect('equal');ax.axis('off')
def row(y0,title,connected):
    ax.add_patch(plt.Rectangle((-0.5,y0-0.95),8.9,1.9,fc=OPEN,ec='none',zorder=0))
    if connected:
        x=np.linspace(-0.5,8.4,400);half=0.22+0.15*np.abs(np.sin(np.pi*x/2))
        ax.fill_between(x,y0-half,y0+half,color=GOLD,lw=0,zorder=1)
    else:
        for xc in np.arange(1,8,2.0): ax.add_patch(Ellipse((xc,y0),1.15,0.72,fc=GOLD,ec='none',zorder=1))
    for xc in np.arange(0,9,2.0):
        for s in (1,-1): ax.add_patch(Circle((xc,y0+s*0.95),0.62,fc=ATOM,ec=EDGE,lw=2,zorder=2))
    ax.add_patch(Circle((4.0,y0),0.42,fc='none',ec=INK,lw=3,ls=(0,(2,1.6)),zorder=3))
    ax.text(-0.5,y0+1.93,title,ha='left',va='center',fontsize=27,fontweight='bold')
    ax.annotate('ELF high at necks' if connected else 'ELF low at necks',xy=(4.45,y0),xytext=(9.0,y0),fontsize=26,
                fontweight='bold',color=INK,va='center',arrowprops={'arrowstyle':'-|>','color':INK,'lw':2.4,'mutation_scale':22})
    ax.text(9.0,y0-0.62,'→ 1D, connected' if connected else '→ 0D, isolated',fontsize=24,color=MUTED,va='center')
row(0.55,'Electrons fill the channel',True)
row(-3.15,'Electrons sit in separate pockets',False)
for ext in ('.png','.pdf'): fig.savefig(OUT+ext,dpi=300,facecolor=BG)
print('saved')
