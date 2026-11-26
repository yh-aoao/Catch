from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import xml.etree.ElementTree as X
base=Path('/home/isee/Catch/analysis_artifacts');d=base/'two_stage_module_20261126/output'
A='http://schemas.openxmlformats.org/drawingml/2006/main';P='http://schemas.openxmlformats.org/presentationml/2006/main';ns={'a':A,'p':P};E=9525
X.register_namespace('a',A);X.register_namespace('p',P)
source=d/'论文模型图_第四页_AB分区精简版.pptx'
with ZipFile(source) as z:root=X.fromstring(z.read('ppt/slides/slide5.xml'))
tree=root.find('p:cSld/p:spTree',ns)
def geom(el,x=None,y=None,w=None,h=None,group=False):
 xf=el.find(('p:grpSpPr' if group else 'p:spPr')+'/a:xfrm',ns)
 for tag,key,value in [('off','x',x),('off','y',y),('ext','cx',w),('ext','cy',h)]:
  if value is not None:xf.find('a:'+tag,ns).set(key,str(round(value*E)))
for g in tree.findall('p:grpSp',ns):
 ident=int(g.find('p:nvGrpSpPr/p:cNvPr',ns).get('id'))
 if ident not in (1000,56):continue
 geom(g,x=55,y=105 if ident==1000 else 990,w=620,h=250,group=True)
 for pr in g.findall('.//a:rPr',ns):
  old=int(pr.get('sz','850'))
  pr.set('sz','1800' if old>=1400 else ('1100' if pr.get('baseline') else '1150'))
  pr.set('b','1');pr.set('lang','en-US')
  for f in pr.findall('a:latin',ns):f.set('typeface','Cambria')
 for ln in g.findall('.//a:ln',ns):ln.set('w','11430')
 for body in g.findall('.//a:bodyPr',ns):
  body.set('wrap','none');body.set('lIns','0');body.set('rIns','0');body.set('tIns','0');body.set('bIns','0')
for el in list(tree)[2:]:
 pr=el.find('.//p:cNvPr',ns)
 if pr is None:continue
 ident=int(pr.get('id'))
 if ident==9002:geom(el,x=55,w=620)
 if ident==9003:geom(el,x=55,y=946,w=235,h=26)
 if ident==9004:geom(el,y=355,h=65)
 if ident==9005:geom(el,y=888,h=102)
 if ident==6030:
  geom(el,y=1230,h=18)
  for rp in el.findall('.//a:rPr',ns):rp.set('sz','1000')
# Native outer frames are inserted behind all existing objects.
def frame(ident,name,x,y,w,h,color,dash):
 sp=X.Element('{'+P+'}sp');nv=X.SubElement(sp,'{'+P+'}nvSpPr');X.SubElement(nv,'{'+P+'}cNvPr',{'id':str(ident),'name':name});X.SubElement(nv,'{'+P+'}cNvSpPr');X.SubElement(nv,'{'+P+'}nvPr')
 pr=X.SubElement(sp,'{'+P+'}spPr');xf=X.SubElement(pr,'{'+A+'}xfrm');X.SubElement(xf,'{'+A+'}off',{'x':str(x*E),'y':str(y*E)});X.SubElement(xf,'{'+A+'}ext',{'cx':str(w*E),'cy':str(h*E)})
 geo=X.SubElement(pr,'{'+A+'}prstGeom',{'prst':'roundRect'});av=X.SubElement(geo,'{'+A+'}avLst');X.SubElement(av,'{'+A+'}gd',{'name':'adj','fmla':'val 4000'})
 X.SubElement(pr,'{'+A+'}noFill');ln=X.SubElement(pr,'{'+A+'}ln',{'w':'18000'});f=X.SubElement(ln,'{'+A+'}solidFill');X.SubElement(f,'{'+A+'}srgbClr',{'val':color})
 if dash:X.SubElement(ln,'{'+A+'}prstDash',{'val':'dash'})
 return sp
# Future includes prediction, projection and residual fusion, but not action heads.
tree.insert(2,frame(11001,'Future displacement module',955,390,480,222,'85A373',True))
tree.insert(2,frame(11002,'Auxiliary supervision module',840,900,1120,327,'C5956D',False))
# Small frame caption uses the same native font as the full figure.
sp=frame(11003,'Future module caption',1000,376,350,28,'FFFFFF',False)
pr=sp.find('p:spPr',ns)
for c in list(pr):
 if c.tag in ('{'+A+'}noFill','{'+A+'}ln'):pr.remove(c)
f=X.SubElement(pr,'{'+A+'}solidFill');X.SubElement(f,'{'+A+'}srgbClr',{'val':'FFFFFF'});ln=X.SubElement(pr,'{'+A+'}ln');X.SubElement(ln,'{'+A+'}noFill')
b=X.SubElement(sp,'{'+P+'}txBody');X.SubElement(b,'{'+A+'}bodyPr',{'anchor':'ctr','wrap':'none','lIns':'0','rIns':'0','tIns':'0','bIns':'0'});X.SubElement(b,'{'+A+'}lstStyle');p=X.SubElement(b,'{'+A+'}p');X.SubElement(p,'{'+A+'}pPr',{'algn':'ctr'});r=X.SubElement(p,'{'+A+'}r');rp=X.SubElement(r,'{'+A+'}rPr',{'lang':'en-US','sz':'1500','b':'1'});ff=X.SubElement(rp,'{'+A+'}solidFill');X.SubElement(ff,'{'+A+'}srgbClr',{'val':'617E51'});X.SubElement(rp,'{'+A+'}latin',{'typeface':'Cambria'});X.SubElement(r,'{'+A+'}t').text='Future Displacement';tree.append(sp)
outpath=d/'论文模型图_第五页_RL统一与模块分组版.pptx'
with ZipFile(source) as z,ZipFile(outpath,'w',ZIP_DEFLATED) as out:
 for item in z.infolist():out.writestr(item.filename,X.tostring(root,encoding='utf-8',xml_declaration=True) if item.filename=='ppt/slides/slide5.xml' else z.read(item.filename))
with ZipFile(source) as z,ZipFile(outpath) as zz:
 assert all(z.read(f'ppt/slides/slide{i}.xml')==zz.read(f'ppt/slides/slide{i}.xml') for i in range(1,5))
print(outpath)
