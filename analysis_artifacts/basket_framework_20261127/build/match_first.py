import copy,zipfile,xml.etree.ElementTree as E
from pathlib import Path
P='http://schemas.openxmlformats.org/presentationml/2006/main';A='http://schemas.openxmlformats.org/drawingml/2006/main';ns={'p':P,'a':A}
E.register_namespace('p',P);E.register_namespace('a',A)
q=lambda n:'{'+ns[n.split(':')[0]]+'}'+n.split(':')[1]
src=Path('analysis_artifacts/two_stage_module_20261126/output/两阶段强化学习.pptx')
with zipfile.ZipFile(src) as z:root=E.fromstring(z.read('ppt/slides/slide1.xml'))
group=root.find('.//p:grpSp',ns)
shapes={''.join(t.text or '' for t in x.findall('.//a:t',ns)):x for x in group.findall('p:sp',ns)}
plain=shapes['Tracking Policy'];symbols=shapes['πhand'];obs=shapes['Oarmt, t−1'];act=shapes['Aarmt'];bg=[x for x in group.findall('p:sp',ns) if not ''.join(t.text or '' for t in x.findall('.//a:t',ns))]
# Retain the original reusable group transform, colors, Cambria font, thin strokes and rounded geometry.
for el in list(group)[2:]:group.remove(el)
idx=10
def box(template,x,y,w,h,text=None,sub=None,size=None):
 global idx
 el=copy.deepcopy(template);idx+=1;el.find('p:nvSpPr/p:cNvPr',ns).set('id',str(idx))
 xf=el.find('p:spPr/a:xfrm',ns);xf.find('a:off',ns).attrib.update(x=str(round(x*9525)),y=str(round(y*9525)));xf.find('a:ext',ns).attrib.update(cx=str(round(w*9525)),cy=str(round(h*9525)))
 if template is plain:
  pr=el.find('p:spPr',ns)
  for ff in list(pr):
   if ff.tag in (q('a:solidFill'),q('a:noFill')):pr.remove(ff)
  E.SubElement(pr,q('a:noFill'))
 if text is not None:
  tx=el.find('p:txBody',ns);pp=tx.find('a:p',ns);r=pp.find('a:r',ns);rp=copy.deepcopy(r.find('a:rPr',ns)) if r is not None else E.Element(q('a:rPr'))
  if size:rp.set('sz',str(round(size*75)))
  rp.attrib.pop('baseline',None)
  for child in list(pp):
   if child.tag!=q('a:pPr'):pp.remove(child)
  rr=E.SubElement(pp,q('a:r'));rr.append(copy.deepcopy(rp));E.SubElement(rr,q('a:t')).text=text
  if sub:
   rr=E.SubElement(pp,q('a:r'));s=copy.deepcopy(rp);s.set('sz',str(round(int(s.get('sz','1500'))*.6)));s.set('baseline','-25000');rr.append(s);E.SubElement(rr,q('a:t')).text=sub
 group.append(el);return el

def edge(points,dashed=False):
 global idx
 # Native line segments, final segment has arrowhead. Source and target follow the same thin line style.
 for k,((x,y),(xx,yy)) in enumerate(zip(points,points[1:])):
  idx+=1;e=E.SubElement(group,q('p:cxnSp'));nv=E.SubElement(e,q('p:nvCxnSpPr'));E.SubElement(nv,q('p:cNvPr'),id=str(idx),name='flow');E.SubElement(nv,q('p:cNvCxnSpPr'));E.SubElement(nv,q('p:nvPr'));pr=E.SubElement(e,q('p:spPr'));xf=E.SubElement(pr,q('a:xfrm'))
  if yy<y:xf.set('flipV','1')
  E.SubElement(xf,q('a:off'),x=str(round(min(x,xx)*9525)),y=str(round(min(y,yy)*9525)));E.SubElement(xf,q('a:ext'),cx=str(round(abs(xx-x)*9525)),cy=str(round(abs(yy-y)*9525)));g=E.SubElement(pr,q('a:prstGeom'),prst='line');E.SubElement(g,q('a:avLst'));ln=E.SubElement(pr,q('a:ln'),w='11430');E.SubElement(E.SubElement(ln,q('a:solidFill')),q('a:srgbClr'),val='283B48')
  if dashed:E.SubElement(ln,q('a:prstDash'),val='dash')
  if k==len(points)-2:E.SubElement(ln,q('a:tailEnd'),type='triangle',w='sm',len='sm')
# Dot legends directly reuse the source slide's shapes.
box(bg[0],20,9,9,9);box(shapes['Tracking Only'],34,3,230,22,'① Fixed Throwing',size=11)
box(bg[1],275,9,9,9);box(shapes['Tracking & Catching'],289,3,280,22,'② Carrying  →  ③ Bridging',size=11)
box(shapes['Observations'],20,34,135,24,'Observations',size=12);box(shapes['Actions'],520,34,135,24,'Actions',size=12)
# Same outer/inner policy framing as slide one.
box(bg[2],240,44,220,260)
box(shapes['Catching Policy'],245,54,210,25,'②–③ Mobile Policy',size=12)
box(bg[3],254,170,192,112)
box(plain,258,176,184,24,'① Throwing Policy',size=11)
box(symbols,275,98,150,32,'π','carry',size=23)
box(shapes['πarm'],275,220,150,32,'π','throw',size=23)
# Shared actor feature path; mode selection connects both full-action heads.
box(plain,172,151,60,25,'f','θ',size=20)
box(plain,468,157,42,25,'c','t',size=20)
# Four observation components and three action components, matching the first slide.
for y,sub in [(93,'robot'),(153,'ball'),(213,'goal'),(273,'prev')]:
 box(obs,20,y,135,32,'O',sub,size=23)
 edge([(155,y+16),(164,y+16),(164,164),(172,164)])
edge([(232,164),(260,164),(260,114),(275,114)])
edge([(260,164),(260,236),(275,236)])
edge([(425,114),(463,114),(463,169),(468,169)])
edge([(425,236),(463,236),(463,169),(468,169)])
for y,sub in [(93,'hand'),(194,'arm'),(247,'base')]:
 box(act,520,y,135,32,'A',sub,size=23)
 edge([(510,169),(515,169),(515,y+16),(520,y+16)])
box(plain,240,315,220,23,'c: carry → throw (rule)',size=10)
# Shift observation bus behind the shapes so it cannot cross labels (all paths remain outside boxes).
dst=Path('analysis_artifacts/basket_framework_20261127/output/两阶段强化学习_Basket同款样式.pptx')
with zipfile.ZipFile(src) as a,zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED) as b:
 for n in a.infolist():b.writestr(n,E.tostring(root,encoding='utf-8',xml_declaration=True) if n.filename=='ppt/slides/slide2.xml' else a.read(n.filename))
with zipfile.ZipFile(src) as a,zipfile.ZipFile(dst) as b:assert [n for n in a.namelist() if a.read(n)!=b.read(n)]==['ppt/slides/slide2.xml']
print(dst)
