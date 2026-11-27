from pathlib import Path
import zipfile,xml.etree.ElementTree as E
P='http://schemas.openxmlformats.org/presentationml/2006/main';A='http://schemas.openxmlformats.org/drawingml/2006/main'
E.register_namespace('p',P);E.register_namespace('a',A)
def q(n):return '{'+(P if n.startswith('p:') else A)+'}'+n.split(':')[1]
def add(parent,n,**kw):return E.SubElement(parent,q(n),{k:str(v) for k,v in kw.items()})
root=E.Element(q('p:sld'));cs=add(root,'p:cSld');tree=add(cs,'p:spTree')
nv=add(tree,'p:nvGrpSpPr');add(nv,'p:cNvPr',id=1,name='');add(nv,'p:cNvGrpSpPr');add(nv,'p:nvPr');add(tree,'p:grpSpPr')
navy='283B48';blue='B8DEEC';pale='F2F8FB';orange='F1D4BF';opale='FCF6F0';bl='6DA6BF';ol='C99D79';idx=1
# Geometry uses the source slide size, 1060 x 570 CSS pixels.
def box(x,y,w,h,text='',size=16,fill=None,line=None,bold=False,align='ctr',name=None):
 global idx
 idx+=1;s=add(tree,'p:sp');nv=add(s,'p:nvSpPr');add(nv,'p:cNvPr',id=idx,name=name or text.replace('\n',' ')[:70]);add(nv,'p:cNvSpPr');add(nv,'p:nvPr');pr=add(s,'p:spPr');xf=add(pr,'a:xfrm');add(xf,'a:off',x=round(x*9525),y=round(y*9525));add(xf,'a:ext',cx=round(w*9525),cy=round(h*9525));g=add(pr,'a:prstGeom',prst='rect');add(g,'a:avLst')
 if fill:add(add(pr,'a:solidFill'),'a:srgbClr',val=fill)
 else:add(pr,'a:noFill')
 ln=add(pr,'a:ln',w=12700)
 if line:add(add(ln,'a:solidFill'),'a:srgbClr',val=line)
 else:add(ln,'a:noFill')
 tx=add(s,'p:txBody');add(tx,'a:bodyPr',anchor='ctr',lIns=55000,rIns=55000,tIns=25000,bIns=25000);add(tx,'a:lstStyle')
 for row in text.split('\n'):
  pp=add(tx,'a:p');add(pp,'a:pPr',algn=align);r=add(pp,'a:r');rp=add(r,'a:rPr',lang='en-US',sz=round(size*75),b='1' if bold else '0');add(add(rp,'a:solidFill'),'a:srgbClr',val=navy);add(rp,'a:latin',typeface='Cambria');add(rp,'a:ea',typeface='Cambria');add(r,'a:t').text=row
 return s

def arrow(x1,y1,x2,y2,color=navy,dash=False):
 global idx
 idx+=1;s=add(tree,'p:cxnSp');nv=add(s,'p:nvCxnSpPr');add(nv,'p:cNvPr',id=idx,name='Data flow' if not dash else 'Control signal');add(nv,'p:cNvCxnSpPr');add(nv,'p:nvPr');pr=add(s,'p:spPr');xf=add(pr,'a:xfrm',**({'flipV':'1'} if y2<y1 else {}));add(xf,'a:off',x=round(min(x1,x2)*9525),y=round(min(y1,y2)*9525));add(xf,'a:ext',cx=round(abs(x2-x1)*9525),cy=round(abs(y2-y1)*9525));g=add(pr,'a:prstGeom',prst='line');add(g,'a:avLst');ln=add(pr,'a:ln',w=18000);add(add(ln,'a:solidFill'),'a:srgbClr',val=color)
 if dash:add(ln,'a:prstDash',val='dash')
 add(ln,'a:tailEnd',type='triangle',w='sm',len='sm')
box(32,16,996,36,'Basket Reinforcement Learning',27,bold=True)
box(32,56,996,23,'Three training steps · Two execution modes',17)
box(32,91,996,24,'Training procedure',18,bold=True,align='l')
for x,title,body,fill,line in [(32,'1  Fixed-base throwing','Randomized physical holds\nLearn throwing from scratch',pale,bl),(392,'2  Carrying curriculum','Longer carrying distances\nFixed-base throw replay',pale,bl),(752,'3  Carry–throw bridging','Carry first, then throw\nContinuous physical state',opale,ol)]:
 box(x,122,276,78,'',fill=fill,line=line);box(x+5,125,266,25,title,18,bold=True);box(x+5,152,266,44,body,16)
arrow(308,162,392,162);arrow(668,162,752,162)
box(309,132,81,25,'Initialize',14);box(669,132,81,25,'Initialize',14)
box(32,205,996,23,'Steps 2–3 continue training all parameters with PPO',16)
box(32,239,996,25,'Final policy architecture',18,bold=True,align='l')
box(32,304,194,105,'Observations (66D)\nRobot · ball · target\nPrevious action\nPhase cue',17,fill=pale,line=bl,bold=False)
box(272,326,166,61,'Shared Actor\nBackbone',19,fill=blue,line=bl,bold=True)
box(482,289,178,48,'Carrying Head\nπ_carry',18,fill=blue,line=bl,bold=True)
box(482,379,178,48,'Throwing Head\nπ_throw',18,fill=orange,line=ol,bold=True)
box(710,326,130,61,'Action\nSelection',18,fill=opale,line=ol,bold=True)
box(883,304,145,105,'Actions (20D)\nBase: 2\nArm: 6\nHand: 12',17,fill=pale,line=bl)
arrow(226,356,272,356);arrow(438,346,482,313);arrow(438,367,482,403);arrow(660,313,710,345);arrow(660,403,710,368);arrow(840,356,883,356)
box(693,266,165,42,'Rule-based phase switch\nCarry → Throw',16,bold=True)
arrow(775,308,775,326,dash=True)
box(32,449,194,40,'Critic V(o, phase)',18,fill=pale,line=bl,bold=True)
arrow(129,409,129,449)
box(253,446,775,45,'Switch when position and hold stability satisfy the rules.\nNo reset of ball, hand, arm or base state at the switch.',16,align='l')
box(32,515,996,25,'Each action head controls the base, arm and hand jointly; the Critic is used during training.',15)
cm=add(root,'p:clrMapOvr');add(cm,'a:masterClrMapping')
src=Path('analysis_artifacts/two_stage_module_20261126/output/两阶段强化学习.pptx');dst=Path('analysis_artifacts/basket_framework_20261127/output/两阶段强化学习_Basket框架更新.pptx')
with zipfile.ZipFile(src) as zin,zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED) as zout:
 for item in zin.infolist():zout.writestr(item,E.tostring(root,encoding='utf-8',xml_declaration=True) if item.filename=='ppt/slides/slide2.xml' else zin.read(item.filename))
with zipfile.ZipFile(src) as a,zipfile.ZipFile(dst) as b:
 changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
 assert changed==['ppt/slides/slide2.xml'],changed
print(dst);print('Verified: only slide2.xml changed; all other parts are byte-identical.')
