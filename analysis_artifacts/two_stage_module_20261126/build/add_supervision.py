from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import xml.etree.ElementTree as X
base=Path('/home/isee/Catch/analysis_artifacts');d=base/'two_stage_module_20261126/output'
src=(base/'paper_overview_20261125/build/create.py').read_text();exec(src[:src.index("blue='E5EEF8'")]);sh.clear();nodes.clear();edges.clear()
A='http://schemas.openxmlformats.org/drawingml/2006/main';P='http://schemas.openxmlformats.org/presentationml/2006/main';ns={'a':A,'p':P}
X.register_namespace('a',A);X.register_namespace('p',P)
navy='283B48';orange='B5743C';pale='FFF6ED';line='C5956D'
source=d/'论文模型图_第四页_完整纵向蒸馏版.pptx'
with ZipFile(source) as z:root=X.fromstring(z.read('ppt/slides/slide4.xml'))
tree=root.find('p:cSld/p:spTree',ns)
# Compact only the right policy vertically to leave a connected supervision band below.
for el in list(tree)[2:]:
 xf=el.find('p:spPr/a:xfrm',ns)
 if xf is None:xf=el.find('p:grpSpPr/a:xfrm',ns)
 if xf is None or int(xf.find('a:off',ns).get('x'))<7400000:continue
 off=xf.find('a:off',ns);ext=xf.find('a:ext',ns)
 off.set('y',str(round(int(off.get('y'))*.7)));ext.set('cy',str(round(int(ext.get('cy'))*.7)))
 txt=''.join(t.text or '' for t in el.findall('.//a:t',ns))
 if 'Arm: end-effector' in txt:
  off.set('x',str(1280*E));off.set('y',str(665*E));ext.set('cx',str(290*E))
 if len(el.findall('p:txBody/a:p',ns))>=2:
  ext.set('cy',str(max(int(ext.get('cy')),62*E)))
 if txt.startswith('Select by c'):
  body=el.find('p:txBody',ns)
  for p in list(body.findall('a:p',ns)):body.remove(p)
  for textvalue in ['Select head by c','u_t = u_t^C or u_t^B']:
   p=X.SubElement(body,'{'+A+'}p');X.SubElement(p,'{'+A+'}pPr',{'algn':'ctr'});r=X.SubElement(p,'{'+A+'}r');pr=X.SubElement(r,'{'+A+'}rPr',{'sz':'1500','b':'1','lang':'en-US'});X.SubElement(pr,'{'+A+'}latin',{'typeface':'Cambria'});X.SubElement(r,'{'+A+'}t').text=textvalue
 if txt.startswith('a') and 'Base / Arm / Hand' in txt:
  body=el.find('p:txBody',ns)
  first=body.find('a:p',ns)
  for c in list(first):
   if c.tag!='{'+A+'}pPr':first.remove(c)
  r=X.SubElement(first,'{'+A+'}r');pr=X.SubElement(r,'{'+A+'}rPr',{'sz':'1700','b':'1','lang':'en-US'});X.SubElement(pr,'{'+A+'}latin',{'typeface':'Cambria'});X.SubElement(r,'{'+A+'}t').text='a_t = tanh(u_t)'
# Multi-expert labels. The bus leads to task matching, not policy averaging.
for y in (444,584,724,864):edge([(645,y),(710,y)],True,orange,False)
edge([(710,444),(710,864)],True,orange,False)
edge([(710,685),(755,685)],True,orange)
shape(755,640,230,95,'Query teacher for task k\nAlign control space\na_t^T',pale,line,14,navy,True)
edge([(870,735),(870,790)],True,orange)
shape(755,790,320,70,'Action Imitation Loss\nL_imit(u_t, a_t^T)',pale,line,18,navy,True)
# Selected student raw outputs enter imitation loss before tanh.
edge([(1300,778),(1200,778),(1200,825),(1075,825)],True,orange)
text(1080,787,115,30,'u_t',17,orange,True,'ctr')
# Two auxiliary heads share the same feature source; losses have different task masks.
text(850,912,1100,30,'Training-only Auxiliary Supervision',21,orange,True,'ctr')
shape(870,970,420,62,'Object Position Head (Catch)\np̂_t^obj ∈ R^3',pale,line,18,navy,True)
shape(1500,970,420,62,'Basket Position Head (Basket)\np̂_t^basket ∈ R^3',pale,line,18,navy,True)
# Explicit shared h_t source from GRU to both auxiliary predictors.
edge([(1535,322),(1980,322),(1980,950),(1080,950),(1080,970)],True,orange)
edge([(1710,950),(1710,970)],True,orange)
text(1800,915,160,30,'h_t',17,orange,True,'ctr')
edge([(1080,1032),(1080,1070)],True,orange)
edge([(1710,1032),(1710,1070)],True,orange)
shape(870,1070,420,65,'Object Position Loss  L_obj\nCatch samples only',pale,line,18,navy,True)
shape(1500,1070,420,65,'Basket Position Loss  L_basket\nValid Basket samples only',pale,line,18,navy,True)
shape(870,1170,420,48,'Simulator target  p_t^obj', 'F4F5F6','A5ADB5',17,navy,True)
shape(1500,1170,420,48,'Simulator relative target  p_t^basket','F4F5F6','A5ADB5',17,navy,True)
edge([(1080,1170),(1080,1135)],True,orange)
edge([(1710,1170),(1710,1135)],True,orange)
text(820,1220,1150,27,'Dashed: training only. Losses optimize the shared encoder / GRU; target states are not policy inputs.',14,orange,True,'ctr')
# Enlarge native mathematical indices, consistent with the existing diagram.
def parse(s):return X.fromstring('<root xmlns:p="'+P+'" xmlns:a="'+A+'">'+s+'</root>')[0]
def rr(p,t,size=1550,base=0,italic=False):
 r=X.SubElement(p,'{'+A+'}r');at={'sz':str(size),'b':'1','lang':'en-US'}
 if base:at['baseline']=str(base)
 if italic:at['i']='1'
 pr=X.SubElement(r,'{'+A+'}rPr',at);f=X.SubElement(pr,'{'+A+'}solidFill');X.SubElement(f,'{'+A+'}srgbClr',{'val':navy});X.SubElement(pr,'{'+A+'}latin',{'typeface':'Cambria'});X.SubElement(r,'{'+A+'}t').text=t

def sym(p,t,sub='',sup='',size=1550):
 rr(p,t,size,italic=True)
 if sub:rr(p,sub,round(size*.85),-25000,True)
 if sup:rr(p,sup,round(size*.85),35000,True)
def formula(p,txt):
 if txt=='a_t^T':sym(p,'a','t','T',1850)
 elif txt=='u_t':sym(p,'u','t',size=1700)
 elif txt=='h_t':sym(p,'h','t',size=1700)
 elif txt.startswith('L_imit'):
  sym(p,'ℒ','imit');rr(p,'(');sym(p,'u','t');rr(p,', ');sym(p,'a','t','T');rr(p,')')
 elif txt.startswith('p̂_t'):
  sym(p,'p̂','t','basket' if 'basket' in txt else 'obj',1600);rr(p,' ∈ ℝ',1600);rr(p,'3',1360,35000)
 elif txt.startswith('Simulator'):
  rr(p,'Simulator relative target  ' if 'relative' in txt else 'Simulator target  ',1350);sym(p,'p','t','basket' if 'basket' in txt else 'obj',1450)
 elif 'Position Loss  L_' in txt:
  rr(p,'Basket Position Loss  ' if 'Basket' in txt else 'Object Position Loss  ',1450);sym(p,'ℒ','basket' if 'Basket' in txt else 'obj',size=1550)
 elif txt=='u_t = u_t^C or u_t^B':
  sym(p,'u','t',size=1450);rr(p,' = ',1450);sym(p,'u','t','C',1450);rr(p,' or ',1450);sym(p,'u','t','B',1450)
 elif txt=='a_t = tanh(u_t)':sym(p,'a','t',size=1650);rr(p,' = tanh(',1650);sym(p,'u','t',size=1650);rr(p,')',1650)
# Process new shapes and two existing edited formulas.
new=[]
for s in sh:new.append(parse(s))
for el in list(tree)[2:]+new:
 for p in el.findall('p:txBody/a:p',ns):
  txt=''.join(t.text or '' for t in p.findall('.//a:t',ns))
  if not (txt in ['a_t^T','u_t','h_t','u_t = u_t^C or u_t^B','a_t = tanh(u_t)'] or txt.startswith(('L_imit','p̂_t','Simulator')) or 'Position Loss  L_' in txt):continue
  for c in list(p):
   if c.tag!='{'+A+'}pPr':p.remove(c)
  formula(p,txt)
for i,el in enumerate(new,6000):
 for pr in el.findall('.//p:cNvPr',ns):pr.set('id',str(i))
 for pr in el.findall('.//a:rPr',ns):
  pr.set('b','1')
  for f in pr.findall('a:latin',ns):f.set('typeface','Cambria')
 for body in el.findall('.//a:bodyPr',ns):
  body.set('wrap','none');body.set('lIns','19050');body.set('rIns','19050');body.set('tIns','0');body.set('bIns','0')
 for ln in el.findall('.//a:ln',ns):ln.set('w','15000')
 tree.append(el)
outpath=d/'论文模型图_第四页_教师蒸馏与辅助监督版.pptx'
with ZipFile(source) as z,ZipFile(outpath,'w',ZIP_DEFLATED) as out:
 for item in z.infolist():out.writestr(item.filename,X.tostring(root,encoding='utf-8',xml_declaration=True) if item.filename=='ppt/slides/slide4.xml' else z.read(item.filename))
print(outpath)
