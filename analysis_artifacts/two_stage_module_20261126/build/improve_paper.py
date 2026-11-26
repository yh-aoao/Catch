from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from copy import deepcopy
import xml.etree.ElementTree as X
base=Path('/home/isee/Catch/analysis_artifacts');outdir=base/'two_stage_module_20261126/output'
src=(base/'paper_overview_20261125/build/create.py').read_text();exec(src[:src.index("blue='E5EEF8'")]);sh.clear();nodes.clear();edges.clear()
A='http://schemas.openxmlformats.org/drawingml/2006/main';P='http://schemas.openxmlformats.org/presentationml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships';ns={'a':A,'p':P}
for prefix,uri in [('a',A),('p',P),('r',R)]:X.register_namespace(prefix,uri)
navy='283B48';blue='EAF4F9';orange='FCF1E6';line='7597AA'
def parse(s):
 return X.fromstring('<root xmlns:p="'+P+'" xmlns:a="'+A+'" xmlns:r="'+R+'">'+s+'</root>')[0]
source=outdir/'论文模型图.pptx'
with ZipFile(source) as z: original=X.fromstring(z.read('ppt/slides/slide4.xml'))
oldtree=original.find('p:cSld/p:spTree',ns)
pics={e.find('.//p:cNvPr',ns).get('name'):e for e in oldtree.findall('p:pic',ns)}
# Upper explanatory module reconstructed as native, editable shapes.
with ZipFile(outdir/'Two_Stage_RL_Compact_English_Bold.pptx') as z:
 group=X.fromstring(z.read('ppt/slides/slide1.xml')).find('.//p:grpSp',ns)
t=group.find('p:grpSpPr/a:xfrm',ns)
for k,v in [('x',60*E),('y',30*E)]:t.find('a:off',ns).set(k,str(v))
for k,v in [('cx',680*E),('cy',350*E)]:t.find('a:ext',ns).set(k,str(v))
for pr in group.findall('.//a:rPr',ns):pr.set('sz',str(round(int(pr.get('sz','750'))*1.65)))
text(60,0,680,30,'(a) Tracking–Catching RL',20,navy,True,'ctr')
# Four task rows, preserving user-provided illustrations.
newpics=[]
for i,task in enumerate(['Throw','Roll','Bounce','Basket']):
 y=390+i*140
 pic=deepcopy(pics[task+'任务示意']);xf=pic.find('p:spPr/a:xfrm',ns)
 xf.find('a:off',ns).set('x',str(45*E));xf.find('a:off',ns).set('y',str(y*E))
 xf.find('a:ext',ns).set('cx',str(110*E));xf.find('a:ext',ns).set('cy',str(110*E));newpics.append(pic)
 text(40,y+108,125,28,task,20,navy,True,'ctr')
 edge([(155,y+54),(220,y+54)],False,navy)
 shape(220,y+30,250,48,'Two-stage RL  '+('(a)' if i<3 else '(b)'), 'F4F0F5','9B8DA0',20,navy,True)
 edge([(470,y+54),(555,y+54)],False,navy)
 shape(555,y+9,90,90,'','F0EAF6','9B8DA0',geom='ellipse')
 # Native typographic teacher symbol; T is superscript, task is subscript.
 shape(550,y+27,100,52,'π','F0EAF6','F0EAF6',30,navy,True,'rect')
 el=parse(sh[-1]);sppr=el.find('p:spPr',ns)
 for item in list(sppr):
  if item.tag in ('{'+A+'}solidFill','{'+A+'}ln'):sppr.remove(item)
 X.SubElement(sppr,'{'+A+'}noFill')
 ln=X.SubElement(sppr,'{'+A+'}ln');X.SubElement(ln,'{'+A+'}noFill')
 body=el.find('p:txBody',ns)
 for par in body.findall('a:p',ns):body.remove(par)
 par=X.SubElement(body,'{'+A+'}p');X.SubElement(par,'{'+A+'}pPr',{'algn':'ctr'})
 for txt,size,baseline in [('π',2600,None),(task.lower(),1200,'-25000'),('T',1300,'35000')]:
  rr=X.SubElement(par,'{'+A+'}r');attrs={'sz':str(size),'b':'1','i':'1','lang':'en-US'}
  if baseline:attrs['baseline']=baseline
  pr=X.SubElement(rr,'{'+A+'}rPr',attrs);fill=X.SubElement(pr,'{'+A+'}solidFill');X.SubElement(fill,'{'+A+'}srgbClr',{'val':navy});X.SubElement(pr,'{'+A+'}latin',{'typeface':'Cambria'});X.SubElement(rr,'{'+A+'}t').text=txt
 sh[-1]=X.tostring(el,encoding='unicode')
# Basket-specific training inset below the Basket row.
text(55,960,680,28,'(b) Fixed-base–Mobile-base Basket RL',20,navy,True,'ctr')
shape(55,1000,310,154,'',blue,'7CA5BD',geom='roundRect')
shape(435,1000,310,154,'',orange,'C99D79',geom='roundRect')
shape(65,1008,290,28,'Stage 1: Fixed-base Throwing',blue,blue,14,navy,True,'rect')
shape(445,1008,290,28,'Stage 2: Mobile-base Throwing',orange,orange,14,navy,True,'rect')
shape(105,1050,210,40,'π_basket^(1)', 'B8DEEC','7CA5BD',22,navy,True)
shape(485,1050,210,40,'π_basket^T','F1D4BF','C99D79',22,navy,True)
shape(65,1100,290,42,'A_arm, A_hand\nA_base = 0',blue,blue,16,navy,True,'rect')
shape(445,1100,290,42,'A_arm, A_hand, A_base\nCarry, stop and throw',orange,orange,16,navy,True,'rect')
edge([(315,1070),(485,1070)],False,navy)
text(345,1032,110,30,'Initialize',14,navy,True,'ctr')
text(70,1160,660,28,'T: teacher policy     (a), (b): training details',16,navy,True,'ctr')
# Convert basket equations to native sub/superscript text, matching the policy circles.
for j,s in enumerate(sh):
 if 'π_basket' not in s and 'A_arm' not in s:continue
 el=parse(s);body=el.find('p:txBody',ns);txt=''.join(t.text or '' for t in el.findall('.//a:t',ns))
 for par in list(body.findall('a:p',ns)):body.remove(par)
 parts=[]
 if txt.startswith('π_'):parts=[[('π',2200,None),('basket',1150,'-25000'),('(1)' if '^(1)' in txt else 'T',1200,'35000')]]
 else:
  parts=[[('A',1350,None),('arm',950,'-25000'),(', A',1350,None),('hand',950,'-25000')]]
  if 'Carry' in txt:parts[0]+=[(', A',1350,None),('base',950,'-25000')];parts.append([('Carry, stop and throw',1250,None)])
  else:parts.append([('A',1350,None),('base',950,'-25000'),(' = 0',1350,None)])
 for row in parts:
  par=X.SubElement(body,'{'+A+'}p');X.SubElement(par,'{'+A+'}pPr',{'algn':'ctr'})
  for txt,size,baseline in row:
   rr=X.SubElement(par,'{'+A+'}r');attrs={'sz':str(size),'b':'1','lang':'en-US'}
   if baseline:attrs['baseline']=baseline
   pr=X.SubElement(rr,'{'+A+'}rPr',attrs);f=X.SubElement(pr,'{'+A+'}solidFill');X.SubElement(f,'{'+A+'}srgbClr',{'val':navy});X.SubElement(pr,'{'+A+'}latin',{'typeface':'Cambria'});X.SubElement(rr,'{'+A+'}t').text=txt
 sh[j]=X.tostring(el,encoding='unicode')
# Build only slide 4; preserve pages 1–3.
tree=oldtree
for el in list(tree)[2:]:tree.remove(el)
for v in sh:tree.append(parse(v))
tree.append(group)
for pic in newpics:tree.append(pic)
for i,el in enumerate(original.findall('.//p:cNvPr',ns),1):el.set('id',str(i))
# Set all new text to the same serif family and thin diagram strokes.
for el in original.findall('.//a:latin',ns):el.set('typeface','Cambria')
for body in original.findall('.//a:bodyPr',ns):
 body.set('wrap','none');body.set('lIns','19050');body.set('rIns','19050');body.set('tIns','0');body.set('bIns','0')
dst=outdir/'论文模型图_第四页完善版.pptx'
with ZipFile(source) as z,ZipFile(dst,'w',ZIP_DEFLATED) as out:
 for item in z.infolist():out.writestr(item.filename,X.tostring(original,encoding='utf-8',xml_declaration=True) if item.filename=='ppt/slides/slide4.xml' else z.read(item.filename))
print(dst)
