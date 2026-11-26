from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import xml.etree.ElementTree as X
base=Path('/home/isee/Catch/analysis_artifacts');d=base/'two_stage_module_20261126/output'
src=(base/'paper_overview_20261125/build/create.py').read_text();exec(src[:src.index("blue='E5EEF8'")]);sh.clear();edges.clear();nodes.clear()
A='http://schemas.openxmlformats.org/drawingml/2006/main';P='http://schemas.openxmlformats.org/presentationml/2006/main';ns={'a':A,'p':P}
X.register_namespace('a',A);X.register_namespace('p',P)
navy='283B48';fill='EFF4FA';stroke='7990A9'
source=d/'论文模型图_第四页完善版.pptx'
with ZipFile(source) as z:root=X.fromstring(z.read('ppt/slides/slide4.xml'))
tree=root.find('p:cSld/p:spTree',ns)
left=[]
for e in list(tree)[2:]:
 xf=e.find('p:spPr/a:xfrm',ns)
 if xf is None:xf=e.find('p:grpSpPr/a:xfrm',ns)
 if xf is not None and int(xf.find('a:off',ns).get('x'))>=7400000:tree.remove(e)
 else:left.append(X.tostring(e))
text(830,55,1130,45,'Visual Student Encoder',27,navy,True,'ctr')
labels=[('Top Depth', 'I_top,t'),('Wrist Depth','I_wrist,t'),('Proprioception','s_t ∈ R^36'),('Previous Action','a_(t−1) ∈ R^20'),('Task Command','c: Catch / Basket')]
for j,(title,symbol) in enumerate(labels):
 x=845+j*225
 shape(x,155,205,85,title+'\n'+symbol,fill,stroke,19,navy,True)
 if j<2:
  shape(x+15,290,175,65,'CNN',fill,stroke,21,navy,True)
  edge([(x+102.5,240),(x+102.5,290)],False,navy)
  edge([(x+102.5,355),(x+102.5,410)],False,navy)
 else:edge([(x+102.5,240),(x+102.5,410)],False,navy)
shape(845,410,1105,90,'Concatenation\nz_t = [f_top,t, f_wrist,t, s_t, a_(t−1), c]',fill,stroke,21,navy,True)
edge([(1397.5,500),(1397.5,565)],False,navy)
shape(1260,565,275,95,'Shared GRU\nh_t = GRU(z_t, h_(t−1))','E0EAF6',stroke,22,navy,True)
edge([(1397.5,660),(1397.5,725)],False,navy)
shape(1260,725,275,60,'h_t ∈ R^256',fill,stroke,23,navy,True)
# Recurrent state is kept separate from the main top-down path.
edge([(1535,610),(1625,610),(1625,535),(1490,535),(1490,565)],False,stroke)
text(1627,550,170,35,'h_(t−1)',18,navy,True,'l')
# Replace underscore notation with native sub/superscript runs.
def parse(s):return X.fromstring('<root xmlns:p="'+P+'" xmlns:a="'+A+'">'+s+'</root>')[0]
def r(par,txt,base=0,size=1600,italic=False):
 rr=X.SubElement(par,'{'+A+'}r');attrs={'lang':'en-US','sz':str(size),'b':'1'}
 if base:attrs['baseline']=str(base)
 if italic:attrs['i']='1'
 pr=X.SubElement(rr,'{'+A+'}rPr',attrs);ff=X.SubElement(pr,'{'+A+'}solidFill');X.SubElement(ff,'{'+A+'}srgbClr',{'val':navy});X.SubElement(pr,'{'+A+'}latin',{'typeface':'Cambria'});X.SubElement(rr,'{'+A+'}t').text=txt

def sym(par,letter,sub='',sup='',size=1600):
 r(par,letter,size=size,italic=True)
 if sub:r(par,sub,-25000,int(size*.68),True)
 if sup:r(par,sup,35000,int(size*.68),True)

def paragraph(body):
 par=X.SubElement(body,'{'+A+'}p');X.SubElement(par,'{'+A+'}pPr',{'algn':'ctr'});return par
for s in sh:
 el=parse(s);txt=''.join(t.text or '' for t in el.findall('.//a:t',ns));body=el.find('p:txBody',ns)
 if body is not None:
  ps=body.findall('a:p',ns);texts=[''.join(t.text or '' for t in p.findall('.//a:t',ns)) for p in ps]
  for p,txt in zip(ps,texts):
   if not any(k in txt for k in ('I_top','I_wrist','s_t','a_(','z_t','h_t','h_(')):continue
   for child in list(p):
    if child.tag!='{'+A+'}pPr':p.remove(child)
   if txt=='I_top,t':sym(p,'I','t','top')
   elif txt=='I_wrist,t':sym(p,'I','t','wrist')
   elif txt=='s_t ∈ R^36':sym(p,'s','t');r(p,' ∈ ℝ');r(p,'36',35000,1050)
   elif txt=='a_(t−1) ∈ R^20':sym(p,'a','t−1');r(p,' ∈ ℝ');r(p,'20',35000,1050)
   elif txt.startswith('z_t'):
    sym(p,'z','t');r(p,' = [');sym(p,'f','t','top');r(p,', ');sym(p,'f','t','wrist');r(p,', ');sym(p,'s','t');r(p,', ');sym(p,'a','t−1');r(p,', c]')
   elif txt.startswith('h_t ='):
    sym(p,'h','t',size=1450);r(p,' = GRU(',size=1450);sym(p,'z','t',size=1450);r(p,', ',size=1450);sym(p,'h','t−1',size=1450);r(p,')',size=1450)
   elif txt=='h_t ∈ R^256':sym(p,'h','t',size=1850);r(p,' ∈ ℝ',size=1850);r(p,'256',35000,1250)
   elif txt=='h_(t−1)':sym(p,'h','t−1',size=1400)
  for pr in el.findall('.//a:rPr',ns):
   pr.set('b','1');pr.set('lang','en-US')
   for f in pr.findall('a:latin',ns):f.set('typeface','Cambria')
  bp=body.find('a:bodyPr',ns);bp.set('wrap','none');bp.set('lIns','19050');bp.set('rIns','19050');bp.set('tIns','0');bp.set('bIns','0')
 for ln in el.findall('.//a:ln',ns):ln.set('w','15000')
 tree.append(el)
# Keep every pre-existing left element byte-equivalent at element level; only assign IDs to new objects.
maxid=max(int(e.get('id')) for e in root.findall('.//p:cNvPr',ns)[:0]) if False else 2000
for e in list(tree)[2+len(left):]:
 for pr in e.findall('.//p:cNvPr',ns):maxid+=1;pr.set('id',str(maxid))
assert left==[X.tostring(e) for e in list(tree)[2:2+len(left)]]
dst=d/'论文模型图_第四页_纵向英文蒸馏版.pptx'
with ZipFile(source) as z,ZipFile(dst,'w',ZIP_DEFLATED) as out:
 for item in z.infolist():out.writestr(item.filename,X.tostring(root,encoding='utf-8',xml_declaration=True) if item.filename=='ppt/slides/slide4.xml' else z.read(item.filename))
print(dst);print('Left-side elements preserved:',len(left))
