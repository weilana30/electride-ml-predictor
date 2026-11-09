"""Build poster/poster.svg (36 x 48 in) from the text in poster_lib.py and the figures.

WARNING: rebuilding overwrites poster.svg, including any edits made by hand in Inkscape.
    python poster/build/build_poster.py [OUT.svg]
"""
import re,sys
from pathlib import Path
from lxml import etree
sys.path.insert(0,str(Path(__file__).resolve().parent))
import poster_lib as lib
from poster_lib import *
POSTER=str(Path(__file__).resolve().parents[1])+'/'
OUT=sys.argv[1] if len(sys.argv)>1 else POSTER+'poster.svg'
INK='http://www.inkscape.org/namespaces/inkscape'
NATIVE={'basic':('diagrams/electride_basic.svg','electride','Basic electride schematic'),'void':('diagrams/void_connectivity_schematic.svg','voidmap','Void connectivity schematic'),'elf':('diagrams/elf_prediction.svg','elf','ELF prediction workflow')}
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
FIG['comp']=('figures/gnn_model_comparison.png',FIG['comp'][1]);FIG['parity']=('figures/gnn_parity_atom_only.png',FIG['parity'][1]);FIG['flow']=('figures/connectivity_flow.png',FIG['flow'][1]);FIG['pipes']=('figures/pipes_vs_electrons.png',FIG['pipes'][1])
KEY='**Gold:** void regions    **Gray and black:** atoms    Images simplified for clarity'
GOLD='#cfb991';AGED='#8e6f3e';RUSH='#daaa00';T1C,T2C='#f4efe7','#fcf5e1';NAVY='#000000'
def head(p,x,y,w,t,c=NAVY,pt=46): return p.text(x,y,w,t,pt,c,True,lh=1.1)+.16
G1,G2='#f2f2f2','#e6e6e6';EB1,EB2=AGED,RUSH
p=Poster();M=1.1;W=36-2*M;C=16.2;X2=36-M-C;mid=18;cx1=M+C/2;cx2=X2+C/2
p.layer('Title')
y=p.text(M,.72,27.2,TITLE,70,NAVY,True,lh=1.07)+.14
y=p.text(M,y,27,SUB,31,TXT)+.12;p.text(M,y,27,AUTH,25,TXT)
p.box(29.0,.8,2.75,2.75,'LOGO');p.box(32.15,.8,2.75,2.75,'QR CODE')
# Background band
Y=4.9;H=9.5
p.layer('Background band');p.rect(0,Y,36,H,G1,0,'Background band')
y=p.eyebrow(M,Y+.42,'BACKGROUND','#555555')
a=head(p,M,y,11.4,'What is an electride?');ya=p.paras(M,a,6.0,INTRO,27,gap=.2)
p.fig('basic',7.35,a+.08,5.0)
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
p.layer('Tool 1: Predict ELF');p.rect(0,Y-.32,mid,35.95-(Y-.32),T1C,0,'Tool 1 panel');p.rect(0,Y-.32,mid-.0,.16,AGED,0,'Tool 1 color edge')
x=M;p.rect(x,Y-.02,4.78,.52,AGED,.06,'Tool 1 tag');p.eyebrow(x+.18,Y+.06,'TOOL 1  ·  PREDICT ELF','#ffffff');y=Y+.78;y=head(p,x,y,C,T1F,NAVY,41)
y=p.stats(x,y,C,S1,AGED,64)+.1
yy=p.fig('elf',x,y,C)+.1;h=5.6;p.fig('comp',x,yy,h=h);p.fig('parity',x+C-h*1.2259,yy,h=h)
e1=p.paras(x,yy+h+.2,C,P1,26,gap=.1,bullet=AGED)
p.layer('Tool 2: Map voids');p.rect(mid,Y-.32,36-mid,35.95-(Y-.32),T2C,0,'Tool 2 panel');p.rect(mid,Y-.32,36-mid,.16,RUSH,0,'Tool 2 color edge')
x=X2;p.rect(x,Y-.02,7.32,.52,RUSH,.06,'Tool 2 tag');p.eyebrow(x+.18,Y+.06,'TOOL 2  ·  MAP HOW VOIDS CONNECT','#000000');y=Y+.78;y=head(p,x,y,C,T2F,NAVY,41)
C2=36-.5-x;v=p.fig('void',x,y+.2,C2);v=p.text(x,v+.06,C2,VCAP,21,TXT,italic=True)+.45
e2=p.fig('flow',x,v,C2)
# Together band: figure left, take-home middle, next steps right
FIG['pipes']=('figures/pipes_vs_electrons.png',3600/2100)
Y=35.95;H=10.8
p.layer('Together band');p.rect(0,Y,36,H,G1,0,'Together band');p.tri(cx1,Y,T1C,1.3,.42);p.tri(cx2,Y,T2C,1.3,.42)
y=p.eyebrow(M,Y+.55,'TOOLS TOGETHER','#555555');a=head(p,M,y,16.4,T3,NAVY,44)
a=p.text(M,a,15.8,'Same open channel, different electrons: only ELF at the necks tells 1D from 0D.',27,TXT)+.3
e3=p.fig('pipes',M,a,h=Y+H-.45-a)
p.rule(18.0,Y+.6,18.0,Y+H-.6,'#c4bfc0',2.4)
mx=18.7;mw=8.6;b=p.eyebrow(mx,Y+.55,'TAKE-HOME','#000000')
b=p.text(mx,b,mw,TAKE,36,NAVY,True,lh=1.12)+.12;b=p.text(mx,b,mw,TAKE2,26,'#000000')+.55
b=p.text(mx,b,mw,'**What this work adds**',30,NAVY)+.1;e4=p.paras(mx,b,mw,ADD,25,gap=.12,bullet='#000000')
p.rule(27.85,Y+.6,27.85,Y+H-.6,'#c4bfc0',2.4)
rx=28.5;rw=36-M-rx;c=p.eyebrow(rx,Y+.55,'NEXT STEPS','#000000');e5=p.paras(rx,c,rw,NEXT,28,'#000000',gap=.2,bullet='#000000')
# References band
Y=46.75;H=48-Y
p.layer('References band');p.rect(0,Y,36,H,'#e6e6e6',0,'References band')
e6=p.text(M,Y+.42,W,'**REFERENCES**    Becke & Edgecombe (1990)  ·  Burton et al. (2018)  ·  Zhu et al. (2019)  ·  Zeo++ (2012)',24,'#000000')
print('ends tool1',round(e1,2),'tool2',round(e2,2),'/35.6 | fig',round(e3,2),'take',round(e4,2),'next',round(e5,2),'/46.3 | refs',round(e6,2),'/47.8')
p.save(OUT)
t=open(OUT).read()
for a,b in (('#13294b','#000000'),('#142b49','#000000'),('#1a1a1a','#000000'),('#33445c','#333333'),('#c9d3de','#c4bfc0'),('#7d8b9c','#6f727b')): t=t.replace(a,b)
open(OUT,'w').write(t)
