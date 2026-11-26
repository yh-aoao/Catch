from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import xml.etree.ElementTree as X
base=Path('/home/isee/Catch/analysis_artifacts');folder=base/'two_stage_module_20261126/output'
src=(base/'paper_overview_20261125/build/create.py').read_text();exec(src[:src.index("blue='E5EEF8'")]);sh.clear();edges.clear();nodes.clear()
A='http://schemas.openxmlformats.org/drawingml/2006/main';P='http://schemas.openxmlformats.org/presentationml/2006/main';ns={'a':A,'p':P}
X.register_namespace('a',A);X.register_namespace('p',P)
navy='283B48';blue='B8DEEC';pale='F2F8FB';orange='F1D4BF';opale='FCF6F0';bl='6DA6BF';ol='C99D79'
# Native editable diagram at the same physical scale as the reference module.
text(0,0,680,30,'Two-stage RL for Basket',18,navy,True,'ctr')
shape(5,49,300,235,'',pale,bl,geom='roundRect')
shape(375,49,300,235,'',opale,ol,geom='roundRect')
shape(15,57,280,32,'Stage 1: Fixed-base Throwing',pale,pale,14,navy,True,'rect')
shape(385,57,280,32,'Stage 2: Mobile-base Throwing',opale,opale,14,navy,True,'rect')
# Observations and policies shown separately in each training stage.
shape(40,102,230,35,'O_proprio, O_basket',pale,bl,17,navy,True)
shape(410,102,230,35,'O_proprio, O_basket',opale,ol,17,navy,True)
shape(70,165,170,42,'π_basket^(1)',blue,bl,23,navy,True)
shape(440,165,170,42,'π_basket^T',orange,ol,23,navy,True)
edge([(155,137),(155,165)],False,navy)
edge([(525,137),(525,165)],False,navy)
edge([(240,186),(440,186)],False,navy)
text(280,143,120,28,'Initialize',14,navy,True,'ctr')
shape(35,231,240,30,'A_arm, A_hand; A_base = 0',pale,pale,15,navy,True,'rect')
shape(405,231,240,30,'A_arm, A_hand, A_base',opale,opale,15,navy,True,'rect')
edge([(155,207),(155,228)],False,navy)
edge([(525,207),(525,228)],False,navy)
text(15,293,280,24,'Learn to throw with a fixed base',13,navy,True,'ctr')
text(385,293,280,24,'Learn to carry, stop and throw',13,navy,True,'ctr')
text(0,330,680,24,'T: teacher policy     Dashed-free arrows: data flow / parameter initialization',11,navy,True,'ctr')
# Structured mathematical runs for every symbol.
def parse(s):return X.fromstring('<root xmlns:p="'+P+'" xmlns:a="'+A+'">'+s+'</root>')[0]
for i,s in enumerate(sh):
 el=parse(s);txt=''.join(t.text or '' for t in el.findall('.//a:t',ns))
 rows=None
 if txt.startswith('O_proprio'):rows=[[('O',1400,None),('proprio',900,'-25000'),(', O',1400,None),('basket',900,'-25000')]]
 if txt.startswith('π_'):rows=[[('π',1900,None),('basket',1000,'-25000'),('(1)' if '^(1)' in txt else 'T',1100,'35000')]]
 if txt.startswith('A_arm'):
  rows=[[('A',1300,None),('arm',850,'-25000'),(', A',1300,None),('hand',850,'-25000')]]
  if '= 0' in txt:rows[0]+=[('; A',1300,None),('base',850,'-25000'),(' = 0',1300,None)]
  else:rows[0]+=[(', A',1300,None),('base',850,'-25000')]
 if rows:
  body=el.find('p:txBody',ns)
  for par in list(body.findall('a:p',ns)):body.remove(par)
  for row in rows:
   par=X.SubElement(body,'{'+A+'}p');X.SubElement(par,'{'+A+'}pPr',{'algn':'ctr'})
   for txt,size,baseline in row:
    r=X.SubElement(par,'{'+A+'}r');attrs={'lang':'en-US','sz':str(size),'b':'1','i':'1'}
    if baseline:attrs['baseline']=baseline
    pr=X.SubElement(r,'{'+A+'}rPr',attrs);f=X.SubElement(pr,'{'+A+'}solidFill');X.SubElement(f,'{'+A+'}srgbClr',{'val':navy});X.SubElement(pr,'{'+A+'}latin',{'typeface':'Cambria'});X.SubElement(r,'{'+A+'}t').text=txt
 sh[i]=X.tostring(el,encoding='unicode')
# All text is bold and uses Cambria. Match the reference's compact 9 cm width.
with ZipFile(folder/'Two_Stage_RL_Compact_English_Bold.pptx') as z:
 root=X.fromstring(z.read('ppt/slides/slide1.xml'))
 tree=root.find('p:cSld/p:spTree',ns);g=tree.find('p:grpSp',ns)
 for child in list(g)[2:]:g.remove(child)
 for s in sh:g.append(parse(s))
 for pr in g.findall('.//a:rPr',ns):
  pr.set('b','1');pr.set('lang','en-US');pr.set('sz',str(round(int(pr.get('sz','1000'))*.5)))
  for f in pr.findall('a:latin',ns):f.set('typeface','Cambria')
 for body in g.findall('.//a:bodyPr',ns):
  body.set('wrap','none');body.set('lIns','0');body.set('rIns','0');body.set('tIns','0');body.set('bIns','0')
 # Re-use same group position/size; slightly taller source coordinate canvas.
 xf=g.find('p:grpSpPr/a:xfrm',ns);xf.find('a:chExt',ns).set('cy',str(360*E));xf.find('a:ext',ns).set('cy',str(180*E))
 for ln in g.findall('.//a:ln',ns):ln.set('w','11430')
 for j,pr in enumerate(root.findall('.//p:cNvPr',ns),1):pr.set('id',str(j))
 for t in g.findall('.//a:t',ns):
  if t.text and t.text.startswith('T: teacher'):t.text='T: teacher policy     Initialize: transfer policy parameters'
 dst=folder/'Basket_Two_Stage_RL_Compact_English_Bold.pptx'
 with ZipFile(dst,'w',ZIP_DEFLATED) as out:
  for item in z.infolist():out.writestr(item.filename,X.tostring(root,encoding='utf-8',xml_declaration=True) if item.filename=='ppt/slides/slide1.xml' else z.read(item.filename))
print(dst)
