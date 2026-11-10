"""Build poster/poster_v3.svg (36 x 48 in): the original layout (build_poster.py) with the v2 header,
"What is an electride?" block, tool questions, confirmation-set parity plot, sweep-labeled Sankey,
and the v2 closing and reference bands.

WARNING: rebuilding overwrites poster_v3.svg, including any edits made by hand in Inkscape.
    python poster/build/build_poster_v3.py [OUT.svg]
"""
import re,sys
from pathlib import Path
from lxml import etree
sys.path.insert(0,str(Path(__file__).resolve().parent))
import poster_lib as lib
from poster_lib import *
POSTER=str(Path(__file__).resolve().parents[1])+'/'
OUT=sys.argv[1] if len(sys.argv)>1 else POSTER+'poster_v3_rebuild.svg'  # never overwrite the hand-edited poster_v3.svg
INK='http://www.inkscape.org/namespaces/inkscape'
NATIVE={'basic':('diagrams/electride_basic.svg','electride','Basic electride schematic'),'void':('diagrams/void_connectivity_schematic.svg','voidmap','Void connectivity schematic'),'elf':('diagrams/elf_prediction.svg','elf','ELF prediction workflow'),'bottleneck':('diagrams/bottleneck_concept.svg','bn','Closing bottleneck concept')}
ZONE,GOALC,T1C,T2C,TOG,TAKEC='#f1f5fa','#dce7f5','#e9f1fb','#fff1e9','#eef1f6','#dce7f5'
defs=[]
def inline(path,pre,label,x,y,w,h):
    root=etree.parse(path,etree.XMLParser(remove_comments=True)).getroot()
    vb=[float(v) for v in root.get('viewBox').split()];extra='';body=[];loc=[]
    for ch in root:
        if not isinstance(ch.tag,str): continue
        t=etree.QName(ch).localname
        if t in('metadata','namedview','title'): continue
        if t=='defs':
            for d in ch:
                if not isinstance(d.tag,str): continue
                if etree.QName(d).localname=='style': extra='stroke-linejoin:round;stroke-linecap:butt';continue
                loc.append(d)
            continue
        body.append(ch)
    for e in loc+body:
        for q in e.iter():
            if isinstance(q.tag,str) and q.get(f'{{{INK}}}groupmode'): del q.attrib[f'{{{INK}}}groupmode']
    def fix(s):
        s=etree.tostring(s,encoding='unicode');s=re.sub(r' xmlns(:\w+)?="[^"]+"','',s)
        s=re.sub(r'\bid="([^"]+)"',lambda m:f'id="{pre}-{m.group(1)}"',s)
        s=re.sub(r'url\(#([^)]+)\)',lambda m:f'url(#{pre}-{m.group(1)})',s)
        return re.sub(r'href="#([^"]+)"',lambda m:f'href="#{pre}-{m.group(1)}"',s)
    defs.extend(fix(d) for d in loc)
    st=f' style="{extra}"' if extra else ''
    return f'<g id="{pre}" inkscape:label="{label}"{st} transform="translate({x*96:.3f} {y*96:.3f}) scale({w*96/vb[2]:.6f} {h*96/vb[3]:.6f}) translate({-vb[0]} {-vb[1]})">'+''.join(fix(b) for b in body)+'</g>'
class Poster(Page):
    def __init__(s): super().__init__();s.layers=[]
    def layer(s,name):
        s.o=[];s.layers.append((name,s.o))
    def fig(s,k,x,y,w=None,h=None):
        f,a=FIG[k]
        if w is None:w=h*a
        h=w/a
        if k in NATIVE:
            n=NATIVE[k];s.o.append(inline(POSTER+n[0],n[1],n[2],x,y,w,h))
        else:
            s.o.append(f'<image xlink:href="{f}" x="{x*96:.1f}" y="{y*96:.1f}" width="{w*96:.1f}" height="{h*96:.1f}" inkscape:label="{k}"/>')
        return y+h
    def rect(s,x,y,w,h,fill,r=.14,label=''):
        s.o.append(f'<rect x="{x*96:.1f}" y="{y*96:.1f}" width="{w*96:.1f}" height="{h*96:.1f}" rx="{r*96:.1f}" ry="{r*96:.1f}" fill="{fill}" inkscape:label="{label}"/>')
    def tri(s,cx,y,c=NAVY,w=.95,h=.27):
        s.o.append(f'<path d="M{(cx-w/2)*96:.1f} {y*96:.1f}H{(cx+w/2)*96:.1f}L{cx*96:.1f} {(y+h)*96:.1f}Z" fill="{c}" inkscape:label="Flow arrow"/>')
    def eyebrow(s,x,y,t,c):
        px=22*96/72
        s.o.append(f'<text x="{x*96:.1f}" y="{(y+.25)*96:.1f}" style="font-family:Arial;font-weight:bold;font-size:{px:.2f}px;letter-spacing:{px*.13:.2f}px;fill:{c}">{t}</text>')
        return y+.42
    def save(s,fn):
        L=''.join(f'<g id="layer-{i}" inkscape:groupmode="layer" inkscape:label="{n}">'+''.join(o)+'</g>' for i,(n,o) in enumerate(s.layers))
        doc=f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:inkscape="{INK}" xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" width="36in" height="48in" viewBox="0 0 3456 4608" version="1.1"><title>Electride poster</title><sodipodi:namedview id="view" pagecolor="#ffffff" inkscape:document-units="in" units="in"/><defs id="poster-defs">{"".join(defs)}</defs><g id="layer-bg" inkscape:groupmode="layer" inkscape:label="Background"><rect width="3456" height="4608" fill="#ffffff"/></g>{L}</svg>'
        etree.fromstring(doc.encode());open(fn,'w').write(doc)

PB='diagrams/blender/crystals/electride_%s_v2_transparent.png'
for k in ('0d','1d','2d'): FIG[k]=(PB%k,1)
CRY=[c for c in CRY if c[0]!='3d']
FCAP=FCAP.replace(', and 3D is left out.','.')
FIG['comp']=('figures/gnn_model_comparison.png',FIG['comp'][1]);FIG['parity']=('figures/gnn_parity_confirmation.png',2387/1947);FIG['bottleneck']=('diagrams/bottleneck_concept.svg',2112/760)
for _k in ('basic','void','elf'):
    _vb=[float(v) for v in etree.parse(POSTER+NATIVE[_k][0]).getroot().get('viewBox').split()];FIG[_k]=(NATIVE[_k][0],_vb[2]/_vb[3]);FIG['flow']=('figures/connectivity_flow_sweep.png',3870/1822);FIG['pipes']=('figures/pipes_vs_electrons.png',FIG['pipes'][1])
KEY='**Gold:** void regions    **Gray and black:** atoms    Images simplified for clarity'
GOLD='#cfb991';AGED='#8e6f3e';RUSH='#daaa00';T1C,T2C='#f4efe7','#fcf5e1';NAVY='#000000'
def head(p,x,y,w,t,c=NAVY,pt=46): return p.text(x,y,w,t,pt,c,True,lh=1.1)+.16
G1,G2='#f2f2f2','#e6e6e6';EB1,EB2=AGED,RUSH
p=Poster();M=1.1;W=36-2*M;C=16.2;X2=36-M-C;mid=18;cx1=M+C/2;cx2=X2+C/2
p.layer('Title')
y=p.text(M,.7,27.4,'Predicting Void ELF and Mapping Connectivity for Electride Screening',72,NAVY,True,lh=1.06)+.12
y=p.text(M,y,27.4,'Electron localization function (ELF) prediction with an atom–void graph network, and exact geometric void connectivity',30,TXT)+.1
p.text(M,y,27.4,'**Andrew Weiland**  ·  [Advisor name]  ·  [Department], Purdue University  ·  [email]',26,TXT)
p.box(29.4,.75,2.6,2.6,'LOGO');p.box(32.3,.75,2.6,2.6,'QR CODE')
# Background band
Y=4.9;H=9.5
p.layer('Background band');p.rect(0,Y,36,H,G1,0,'Background band')
y=p.eyebrow(M,Y+.42,'BACKGROUND','#555555')
a=head(p,M,y,10.6,'What is an electride?',NAVY,42)
a=p.text(M,a,10.4,'In an electride, extra electrons sit in the empty space between atoms and act as anions. Whether those regions are isolated (0D) or connected (1D, 2D) shapes the material’s properties.',27,TXT,lh=1.22)+.3
p.fig('basic',M+2.35,a,h=Y+H-.35-a)
x2=13.4;w2=36-M-x2;b=head(p,x2,y,w2,'Connectivity defines dimensionality in electrides',NAVY,46)
cw=5.6;gp=(w2-3*cw)/2
for i,(k,l,d) in enumerate(CRY):
    q=x2+i*(cw+gp);p.text(q,b,cw,l,26,NAVY,align='center');yy=p.fig(k,q,b+.46,cw);p.text(q,yy+.02,cw,d.replace('separate cages','separate\u00a0cages'),23,TXT,align='center')
p.text(x2,Y+H-.55,w2,KEY,22,TXT,align='center')
# Goal
p.layer('Goal')
Y=14.4;p.rect(0,Y,36,1.45,'#1c1c1e',0,'Goal band');p.text(M,Y+.45,W,GOAL.replace('**Goal:**','**GOAL**  '),33,'#ffffff',lh=1.18)
p.tri(cx1,Y+1.45,'#1c1c1e',1.3,.42);p.tri(cx2,Y+1.45,'#1c1c1e',1.3,.42)
# Tools on white
Y=16.75
p.layer('Tool 1: Predict ELF');p.rect(0,Y-.32,mid,36.4-(Y-.32),T1C,0,'Tool 1 panel');p.rect(0,Y-.32,mid-.0,.16,AGED,0,'Tool 1 color edge')
x=M;p.rect(x,Y-.02,4.78,.52,AGED,.06,'Tool 1 tag');p.eyebrow(x+.18,Y+.06,'TOOL 1  ·  PREDICT ELF','#ffffff');y=Y+.78;y=head(p,x,y,C,'Can a graph network predict ELF at each candidate void from the surrounding atoms?',NAVY,41)
y=p.stats(x,y,C,S1,AGED,64)+.1
# no bullets: the ELF diagram and the two plots share the column height
ac,ap=FIG['comp'][1],FIG['parity'][1];h=(C-.35)/(ac+ap)
eh=C/FIG['elf'][1];free=36.05-(y+eh+h);g=max(free/3,.1)
yy=p.fig('elf',x,y+g,C)+g;p.fig('comp',x,yy,h=h);p.fig('parity',x+C-h*ap,yy,h=h);e1=yy+h
p.layer('Tool 2: Map voids');p.rect(mid,Y-.32,36-mid,36.4-(Y-.32),T2C,0,'Tool 2 panel');p.rect(mid,Y-.32,36-mid,.16,RUSH,0,'Tool 2 color edge')
x=X2;p.rect(x,Y-.02,7.32,.52,RUSH,.06,'Tool 2 tag');p.eyebrow(x+.18,Y+.06,'TOOL 2  ·  MAP HOW VOIDS CONNECT','#000000');y=Y+.78;y=head(p,x,y,C,'Can geometry alone recover the dimensionality of known electrides?',NAVY,41)
C2=36-.5-x;v=p.fig('void',x,y+.2,C2);v=p.text(x,v+.06,C2,VCAP,21,TXT,italic=True)+.45
e2=p.fig('flow',x,v,C2)
# Closing band (v2)
MUTED,RULE2='#555960','#c4bfc0'
Y=36.4;H=46.15-Y
p.layer('Closing band');p.rect(0,Y,36,H,G1,0,'Closing band');p.tri(cx1,Y,T1C,1.3,.42);p.tri(cx2,Y,T2C,1.3,.42)
y=p.eyebrow(M,Y+.5,'PROPOSED COMBINATION  ·  FUTURE WORK','#555555')
y=head(p,M,y,22.5,'Same open channel, different electrons',NAVY,44)
y=p.text(M,y-.06,22.5,'This is how 50 of 167 0D-labeled structures can show connected open space. Conceptual example of the proposed test: geometry finds the bottleneck, and ELF there would decide 1D vs 0D.',22,MUTED)+.12
dh=Y+H-.3-y;fw=min(23.0,dh*FIG['bottleneck'][1])
e3=p.fig('bottleneck',M,y,fw)
p.rule(25.0,Y+.6,25.0,Y+H-.6,RULE2,2.4)
rx=25.6;rw=36-M-rx
c=p.eyebrow(rx,Y+.5,'TAKEAWAY','#000000')
c=p.text(rx,c,rw,'Geometry shows where a path could exist; ELF shows whether electrons occupy it. Together they can separate structural from electronic dimensionality.',34,NAVY,True,lh=1.15)+.55
c=p.eyebrow(rx,c,'NEXT STEP','#000000')
c=p.text(rx,c,rw,'Predict ELF at bottlenecks as well as void centers, and validate both against DFT on known electrides.',30,TXT)+.3
e4=e5=p.text(rx,c,rw,'Then combine them into a crystal-level electride screen. Neither tool is yet a validated electride classifier.',26,MUTED)
# References band (v2)
Y=46.15
p.layer('References band');p.rect(0,Y,36,48-Y,'#e6e6e6',0,'References band')
REFS=('**References**  [1] Becke & Edgecombe, J. Chem. Phys. 92, 5397 (1990), ELF.  [2] Ellis, Ellis & Miao, J. Mater. Inf. 6, 45 (2026), doi:10.20517/jmi.2026.44, ELFNet.  '
      '[3] Registry labels: Magnetic electrides, J. Am. Chem. Soc. (2023), doi:10.1021/jacs.3c00284; Zhu, Frolov & Choudhary, Matter 1, 1293 (2019); '
      'Burton et al., Chem. Mater. 30, 7521 (2018).  [4] Methods: Shannon, Acta Cryst. A32, 751 (1976), ionic radii; Willems et al., Micropor. Mesopor. Mater. 149, 134 (2012), Zeo++; '
      'Gilmer et al., ICML (2017), message passing.  **Acknowledgements:** [advisor, funding]; Purdue RCAC (Negishi).')
e6=p.text(M,Y+.3,W,REFS,20,'#000000',lh=1.18)
print('ends tool1',round(e1,2),'tool2',round(e2,2),'/35.6 | fig',round(e3,2),'take',round(e4,2),'next',round(e5,2),'/46.3 | refs',round(e6,2),'/47.8')
p.save(OUT)
t=open(OUT).read()
for a,b in (('#13294b','#000000'),('#142b49','#000000'),('#1a1a1a','#000000'),('#33445c','#333333'),('#c9d3de','#c4bfc0'),('#7d8b9c','#6f727b')): t=t.replace(a,b)
open(OUT,'w').write(t)
