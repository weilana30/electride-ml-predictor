import re,html
from PIL import ImageFont
R=''  # figure paths are relative to poster/
B='diagrams/blender/crystals/electride_%s_v2_transparent.png'
FIG={'basic':('diagrams/electride_basic.png',1),'0d':(B%'0d',1),'1d':(B%'1d',1),'2d':(B%'2d',1),'3d':(B%'3d',1),'elf':('diagrams/elf_prediction.png',4800/2113),
 'void':('diagrams/void_connectivity_schematic.png',1800/780),'flow':('figures/connectivity_flow.png',3870/1815),'comp':('figures/gnn_model_comparison.png',2499/1791),'parity':('figures/gnn_parity_atom_only.png',2387/1947),'pipes':('figures/pipes_vs_electrons.png',4140/1680)}
NAVY,BLUE,ORANGE,DOR,DBL,TXT,RULE='#13294b','#2a78d6','#eb6834','#c24a1a','#1f5faf','#33445c','#c9d3de'
FD='/System/Library/Fonts/Supplemental/';_f={}
def font(b,i):
    k=(b,i)
    if k not in _f:_f[k]=ImageFont.truetype(FD+('Arial Bold.ttf' if b else 'Arial Italic.ttf' if i else 'Arial.ttf'),400)
    return _f[k]
def parse(s,bold=False):
    out=[]
    for tok in re.split(r'(\*\*.+?\*\*|~[^~]+~)',s):
        if not tok: continue
        if tok.startswith('**'): out.append((tok[2:-2],True,False))
        elif tok.startswith('~'): out.append((tok[1:-1],bold,True))
        else: out.append((tok,bold,False))
    return out
def lines(s,pt,w,bold=False,italic=False):
    wpx=w*72/pt*400/1 /1  # width in font units at size 400
    cur=0;n=1
    for t,b,sub in parse(s,bold):
        for tok in re.findall(r'\S+\s*|\s+',t):
            f=font(b,italic);full=f.getlength(tok)*(.65 if sub else 1);bare=f.getlength(tok.rstrip())*(.65 if sub else 1)
            if cur+bare>wpx and cur>0:n+=1;cur=0
            cur+=full
    return n
class Page:
    def __init__(s): s.o=[]
    def text(s,x,y,w,t,pt,color=TXT,bold=False,italic=False,align='start',lh=1.2):
        px=pt*96/72;n=lines(t,pt,w,bold,italic);base=(y+pt/72*.93)*96
        ax=(x+w/2)*96 if align=='center' else (x+w if align=='end' else x)*96 if align!='start' else x*96
        an={'start':'start','center':'middle','end':'end'}[align]
        sp=''
        for tk,b,sub in parse(t,bold):
            st=f"font-weight:{'bold' if b else 'normal'};font-style:{'italic' if italic else 'normal'}"
            if sub: st+=';baseline-shift:sub;font-size:65%'
            sp+=f'<tspan style="{st}">{html.escape(tk,quote=False)}</tspan>'
        s.o.append(f'<text xml:space="preserve" x="{ax:.1f}" y="{base:.1f}" style="font-family:Arial;font-size:{px:.2f}px;line-height:{lh};fill:{color};text-anchor:{an};text-align:{align};inline-size:{w*96+2:.1f}px;white-space:pre-wrap">{sp}</text>')
        return y+n*pt/72*lh
    def paras(s,x,y,w,ps,pt,color=TXT,gap=.16,bullet=None):
        for p in ps:
            if bullet:
                s.o.append(f'<circle cx="{(x+.07)*96:.1f}" cy="{(y+pt/72*.56)*96:.1f}" r="{pt*.13*96/72:.1f}" fill="{bullet}"/>')
                y=s.text(x+.34,y,w-.34,p,pt,color)+gap
            else: y=s.text(x,y,w,p,pt,color)+gap
        return y-gap
    def fig(s,k,x,y,w=None,h=None):
        f,a=FIG[k]
        if w is None:w=h*a
        h=w/a
        s.o.append(f'<image xlink:href="{R}{f}" x="{x*96:.1f}" y="{y*96:.1f}" width="{w*96:.1f}" height="{h*96:.1f}"/>')
        return y+h
    def rule(s,x1,y1,x2,y2,c=RULE,w=2.2):
        s.o.append(f'<line x1="{x1*96:.1f}" y1="{y1*96:.1f}" x2="{x2*96:.1f}" y2="{y2*96:.1f}" stroke="{c}" stroke-width="{w}"/>')
    def box(s,x,y,w,h,label):
        s.o.append(f'<rect x="{x*96:.1f}" y="{y*96:.1f}" width="{w*96:.1f}" height="{h*96:.1f}" fill="none" stroke="{RULE}" stroke-width="2.5"/>')
        s.text(x,y+h/2-.16,w,label,17,'#7d8b9c',italic=True,align='center')
    def stats(s,x,y,w,vals,color,npt=66):
        cw=w/len(vals)
        for i,(n,l) in enumerate(vals):
            s.text(x+i*cw,y,cw-.2,n,npt,color,True)
            s.text(x+i*cw,y+npt/72*1.14,cw-.25,l,22,TXT)
        return y+npt/72*1.14+.42
    def save(s,fn):
        open(fn,'w').write(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="36in" height="48in" viewBox="0 0 3456 4608"><rect width="3456" height="4608" fill="#fff"/>{"".join(s.o)}</svg>')
TITLE='Predicting Interstitial Electron Localization and Void Connectivity for Electride Screening'
SUB='A void-centered machine-learning and geometric framework for characterizing electrides'
AUTH='**Andrew Weiland**  ·  [Advisor name]  ·  [Department], Purdue University  ·  [email]'
INTRO=['**A crystal where electrons act as anions.** Extra electrons sit in the voids between atoms.','**ELF marks them.** The electron localization function runs from 0 to 1. A high value in a void means a trapped electron.','**Finding them is slow.** Each crystal needs its own DFT calculation.']
BASICCAP='Blue: cations. Orange: electrons held in the voids, in place of anions.'
CRY=[('0d','**0D**  Isolated pockets','Mayenite-type Ca~6~Al~7~O~16~: separate cages'),('1d','**1D**  Channels','Y~5~Si~3~: channels along one direction'),('2d','**2D**  Sheets','Ca~2~N: layers between the atoms'),('3d','**3D**  Network','Schematic: open in all directions')]
KEY='**Orange:** void regions    **Blue and navy:** atoms    Images simplified for clarity'
GOAL='**Goal:** screen crystals for electrides from structure alone, without DFT on every candidate. We built two void-centered tools and tested each one separately.'
T1='Predict ELF at every void';T1F='A graph neural network predicts ELF at individual voids from the surrounding atoms'
S1=[('0.82','R² on unseen families'),('0.059','RMSE on the 0–1 ELF scale'),('88,680','test voids, 5,000 crystals')]
P1=['**Atoms carry the signal:** a void alone predicts almost nothing (RMSE 0.140).','**Limits:** trained on ELFNet (compressed hydrides), so not yet tested on known electrides. Rare voids with ELF ≥ 0.7 are under-predicted.']
T2='Map how voids connect';T2F='Geometry alone recovers the dimensionality of known electrides above chance'
VCAP='Schematic of an illustrative crystal, not a real material'
FCAP='**Across all probe sizes:** a crystal counts as channels if any probe gives 1D, otherwise sheets, otherwise pockets. Exploratory: this rule was set after seeing the labels, and 3D is left out.'
P2=['**Method:** ionic radius + 1.0 Å probe; an exact power-diagram network counts repeat directions.','**Ionic radii win:** 57% balanced accuracy at 1.0 Å (chance 33%) vs 45% with one size for all atoms.','**Probe size matters:** the right dimension appears at some probe for 99% of structures.','**Main miss at 1.0 Å:** 50 of 167 0D electrides look 2D or 3D. The space connects; the\u00a0electrons\u00a0do\u00a0not.','**Caution:** only 9 of the 260 known electrides are experimentally confirmed.']
T3='Open space is not the same as electrons'
W1=['Geometry asks whether a path exists through open space. ELF asks whether electrons actually\u00a0occupy\u00a0it.','A channel can be electronically empty, while electrons in a pocket stay disconnected.']
ADD=['A prediction target at specific crystal voids','An atom–void graph that keeps chemical and geometric context','An exact connectivity screen for void networks','A path to combining geometry with electronic evidence']
TAKE='Predicting ELF at voids and mapping void connectivity build a screen that can separate structural from electronic dimensionality.'
TAKE2='A foundation, not yet a validated electride classifier.'
NEXT=['Run DFT on known electrides to test the GNN','Predict ELF at void necks','Build a crystal-level electride score']
REFS='Becke & Edgecombe (1990) · Burton et al. (2018) · Zhu et al. (2019) · Zeo++ (2012)'
