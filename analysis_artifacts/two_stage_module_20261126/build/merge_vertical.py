from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import xml.etree.ElementTree as X
base=Path('/home/isee/Catch/analysis_artifacts');d=base/'two_stage_module_20261126/output'
src=(base/'paper_overview_20261125/build/create.py').read_text();exec(src[:src.index("blue='E5EEF8'")]);sh.clear();nodes.clear();edges.clear()
A='http://schemas.openxmlformats.org/drawingml/2006/main';P='http://schemas.openxmlformats.org/presentationml/2006/main';ns={'a':A,'p':P}
X.register_namespace('a',A);X.register_namespace('p',P)
source=d/'论文模型图_第四页_纵向英文蒸馏_大字号版.pptx'
with ZipFile(source) as z:root=X.fromstring(z.read('ppt/slides/slide4.xml'))
tree=root.find('p:cSld/p:spTree',ns);left=[]
for el in list(tree)[2:]:
 ident=int(el.find('.//p:cNvPr',ns).get('id'))
 xf=el.find('p:spPr/a:xfrm',ns)
 if xf is None:xf=el.find('p:grpSpPr/a:xfrm',ns)
 xx=int(xf.find('a:off',ns).get('x')) if xf is not None else 0
 if xx<7400000:left.append(X.tostring(el));continue
 if ident<2000 or ident==2019:tree.remove(el);continue
 # Upper encoder uses less vertical space while preserving font size and all formulas.
 if xf is not None:
  off=xf.find('a:off',ns);ext=xf.find('a:ext',ns)
  off.set('y',str(round(int(off.get('y'))*.75)));ext.set('cy',str(round(int(ext.get('cy'))*.75)))
 for t in el.findall('.//a:t',ns):
  if t.text=='Visual Student Encoder':t.text='Visual Student Policy'
navy='283B48';stroke='7990A9';blue='EFF4FA';green='EAF4E5';gs='85A373';purple='F0EAF6';ps='9B8DA0'
# Trunk feature splits into a future-augmented Catch branch and direct Basket branch.
edge([(1397.5,495),(1397.5,540)],False,navy)
text(1405,510,165,35,'h_t ∈ R^256',19,navy,True,'l')
edge([(1397.5,540),(1140,540),(1140,590)],False,navy)
shape(985,590,310,80,'Future Prediction Head\nΔp_(t+{1,3,5}) ∈ R^9',green,gs,20,navy,True)
edge([(1140,670),(1140,710)],False,navy)
shape(1030,710,220,55,'Linear: 9 → 256',green,gs,19,navy,True)
edge([(1140,765),(1140,805)],False,navy)
shape(1115,805,50,50,'+',green,gs,26,navy,True,geom='ellipse')
edge([(1397.5,540),(1397.5,830),(1165,830)],False,navy)
text(1240,790,145,32,'h_t',18,navy,True,'ctr')
edge([(1397.5,540),(1725,540),(1725,900)],False,navy)
edge([(1140,855),(1140,900)],False,navy)
shape(995,900,290,75,'Catch Action Head\nu_t^C ∈ R^20',green,gs,21,navy,True)
shape(1580,900,290,75,'Basket Action Head\nu_t^B ∈ R^20',purple,ps,21,navy,True)
text(830,980,285,48,'Arm: end-effector\npose increments',15,navy,True,'ctr')
text(1740,980,210,48,'Arm: joint\nvelocities',15,navy,True,'ctr')
edge([(1140,975),(1140,1092),(1300,1092)],False,navy)
edge([(1725,975),(1725,1092),(1600,1092)],False,navy)
shape(1300,1060,300,65,'Select by c + tanh\na_t = tanh(u_t^c)',purple,ps,21,navy,True)
edge([(1450,1125),(1450,1170)],False,navy)
shape(1260,1170,380,65,'a_t ∈ R^20\nBase / Arm / Hand',blue,stroke,22,navy,True)
# Native sub/superscripts, with enlarged indices matching the encoder.
def parse(s):return X.fromstring('<root xmlns:p="'+P+'" xmlns:a="'+A+'">'+s+'</root>')[0]
def rr(par,txt,size=1800,baseline=0,italic=False):
 r=X.SubElement(par,'{'+A+'}r');at={'lang':'en-US','sz':str(size),'b':'1'}
 if baseline:at['baseline']=str(baseline)
 if italic:at['i']='1'
 pr=X.SubElement(r,'{'+A+'}rPr',at);ff=X.SubElement(pr,'{'+A+'}solidFill');X.SubElement(ff,'{'+A+'}srgbClr',{'val':navy});X.SubElement(pr,'{'+A+'}latin',{'typeface':'Cambria'});X.SubElement(r,'{'+A+'}t').text=txt

def sym(p,t,sub='',sup='',size=1800):
 rr(p,t,size,italic=True)
 if sub:rr(p,sub,round(size*.86),-25000,True)
 if sup:rr(p,sup,round(size*.86),35000,True)
def real(p,dim,size=1800):rr(p,' ∈ ℝ',size);rr(p,dim,round(size*.86),35000)
for s in sh:
 el=parse(s);body=el.find('p:txBody',ns)
 if body is not None:
  for p in body.findall('a:p',ns):
   txt=''.join(t.text or '' for t in p.findall('.//a:t',ns))
   if not any(k in txt for k in ('h_t','Δp_','u_t','a_t')):continue
   for c in list(p):
    if c.tag!='{'+A+'}pPr':p.remove(c)
   if txt=='h_t ∈ R^256':sym(p,'h','t');real(p,'256')
   elif txt=='h_t':sym(p,'h','t')
   elif txt.startswith('Δp_'):sym(p,'Δp','t+{1,3,5}',size=1650);real(p,'9',1650)
   elif txt.startswith('u_t'):sym(p,'u','t','C' if '^C' in txt else 'B');real(p,'20')
   elif txt=='a_t = tanh(u_t^c)':sym(p,'a','t',size=1650);rr(p,' = tanh(',1650);sym(p,'u','t','c',1650);rr(p,')',1650)
   elif txt=='a_t ∈ R^20':sym(p,'a','t',size=1950);real(p,'20',1950)
  for pr in el.findall('.//a:rPr',ns):
   pr.set('b','1');pr.set('lang','en-US')
   for f in pr.findall('a:latin',ns):f.set('typeface','Cambria')
  bp=body.find('a:bodyPr',ns);bp.set('wrap','none');bp.set('lIns','19050');bp.set('rIns','19050');bp.set('tIns','0');bp.set('bIns','0')
 for ln in el.findall('.//a:ln',ns):ln.set('w','15000')
 for pr in el.findall('.//p:cNvPr',ns):
  globalid=int(pr.get('id'))+4000;pr.set('id',str(globalid))
 tree.append(el)
# Verify left-side shapes remain unmodified.
current=[]
for el in list(tree)[2:]:
 xf=el.find('p:spPr/a:xfrm',ns)
 if xf is None:xf=el.find('p:grpSpPr/a:xfrm',ns)
 if xf is not None and int(xf.find('a:off',ns).get('x'))<7400000:current.append(X.tostring(el))
assert current==left
outpath=d/'论文模型图_第四页_完整纵向蒸馏版.pptx'
with ZipFile(source) as z,ZipFile(outpath,'w',ZIP_DEFLATED) as out:
 for item in z.infolist():out.writestr(item.filename,X.tostring(root,encoding='utf-8',xml_declaration=True) if item.filename=='ppt/slides/slide4.xml' else z.read(item.filename))
print(outpath)
