from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import xml.etree.ElementTree as X
base=Path('/home/isee/Catch/analysis_artifacts');d=base/'two_stage_module_20261126/output'
src=(base/'paper_overview_20261125/build/create.py').read_text();exec(src[:src.index("blue='E5EEF8'")]);sh.clear();edges.clear();nodes.clear()
A='http://schemas.openxmlformats.org/drawingml/2006/main';P='http://schemas.openxmlformats.org/presentationml/2006/main';ns={'a':A,'p':P}
X.register_namespace('a',A);X.register_namespace('p',P)
source=d/'论文模型图_第四页_教师蒸馏与辅助监督版.pptx'
with ZipFile(source) as z:root=X.fromstring(z.read('ppt/slides/slide4.xml'))
tree=root.find('p:cSld/p:spTree',ns)
navy='283B48';gray='8B98A3';orange='B5743C'
def settext(el,txt,size=1700):
 body=el.find('p:txBody',ns)
 if body is None:return
 for p in list(body.findall('a:p',ns)):body.remove(p)
 for line in txt.split('\n'):
  p=X.SubElement(body,'{'+A+'}p');X.SubElement(p,'{'+A+'}pPr',{'algn':'ctr'})
  r=X.SubElement(p,'{'+A+'}r');pr=X.SubElement(r,'{'+A+'}rPr',{'sz':str(size),'b':'1','lang':'en-US'});f=X.SubElement(pr,'{'+A+'}solidFill');X.SubElement(f,'{'+A+'}srgbClr',{'val':navy});X.SubElement(pr,'{'+A+'}latin',{'typeface':'Cambria'});X.SubElement(r,'{'+A+'}t').text=line
for el in list(tree)[2:]:
 ident=el.find('.//p:cNvPr',ns)
 if ident is None:continue
 ident=int(ident.get('id'))
 if el.tag!='{'+P+'}sp':continue
 txt=''.join(t.text or '' for t in el.findall('.//a:t',ns))
 if ident in (2,27,2001):tree.remove(el);continue
 if ident in (5,11,17):settext(el,'① Catch RL',1900)
 if ident==23:settext(el,'② Basket RL',1900)
 # Smaller type only in the student's upper encoder region.
 xf=el.find('p:spPr/a:xfrm',ns)
 x=int(xf.find('a:off',ns).get('x'))/E if xf is not None else 0
 y=int(xf.find('a:off',ns).get('y'))/E if xf is not None else 0
 if x>800 and y<400:
  for pr in el.findall('.//a:rPr',ns):pr.set('sz',str(round(int(pr.get('sz','1800'))*.78)))
 replacements={'Future Prediction Head':'Future head','Catch Action Head':'Catch head','Basket Action Head':'Basket head','Select head by c':'Select by c','Action Imitation Loss':'Distillation loss','Object Position Head (Catch)':'Object head','Basket Position Head (Basket)':'Basket head','Catch samples only':'Catch','Valid Basket samples only':'Basket','Training-only Auxiliary Supervision':'Auxiliary Supervision (Training Only)'}
 for t in el.findall('.//a:t',ns):
  if t.text in replacements:t.text=replacements[t.text]
 body=el.find('p:txBody',ns)
 if body is None:continue
 if ident==2016:
  settext(el,'Concatenation',1900)
 if ident==6006:
  paragraphs=body.findall('a:p',ns)
  if len(paragraphs)>=3:
   body.remove(paragraphs[1]);t=paragraphs[0].find('.//a:t',ns);t.text='Teacher label'
 if ident==4027:
  for p in list(body.findall('a:p',ns))[1:]:body.remove(p)
 if ident in (6024,6025):
  p=body.find('a:p',ns)
  # Retain only the mathematical loss name, not the repeated prose.
  for r in list(p.findall('a:r',ns)):
   tx=''.join(t.text or '' for t in r.findall('a:t',ns))
   if 'Position Loss' in tx:p.remove(r)
 if ident in (6026,6027):
  for t in el.findall('.//a:t',ns):
   if t.text and t.text.startswith('Simulator'):t.text='Ground truth  '
 if ident==6030:settext(el,'Solid: policy flow     Orange dashed: supervision     Gray dashed: RL detail',1250)
# The two detailed RL insets now connect explicitly to the task-level blocks.
text(20,16,700,43,'(a) Two-stage Reinforcement Learning',26,navy,True,'ctr')
text(830,16,1130,43,'(b) Policy Distillation',26,navy,True,'ctr')
text(104,65,366,30,'① Catch RL',18,navy,True,'ctr')
text(145,978,180,30,'② Basket RL',18,navy,True,'ctr')
edge([(345,303),(345,420)],True,gray)
edge([(345,888),(345,1018)],True,gray)
# Fine divider visually separates the two paper panels.
edge([(737,30),(737,1220)],False,'D5DBE0',False)
# Restore the dimensional alignment if the user removed the earlier Linear caption.
# Existing projection arrows are retained; keep this module compact.
if not any('Linear' in ''.join(t.text or '' for t in e.findall('.//a:t',ns)) for e in tree):
 shape(1030,497,220,40,'Linear','EAF4E5','85A373',17,navy,True)
def parse(s):return X.fromstring('<root xmlns:p="'+P+'" xmlns:a="'+A+'">'+s+'</root>')[0]
for i,s in enumerate(sh,9000):
 el=parse(s)
 for pr in el.findall('.//p:cNvPr',ns):pr.set('id',str(i))
 for pr in el.findall('.//a:rPr',ns):
  pr.set('b','1')
  for f in pr.findall('a:latin',ns):f.set('typeface','Cambria')
 for bp in el.findall('.//a:bodyPr',ns):
  bp.set('wrap','none');bp.set('lIns','19050');bp.set('rIns','19050');bp.set('tIns','0');bp.set('bIns','0')
 tree.append(el)
# Avoid inherited auto-fit from changing the chosen fonts.
for bp in root.findall('.//a:bodyPr',ns):
 bp.set('wrap','none')
 for af in bp.findall('a:normAutofit',ns):bp.remove(af)
dst=d/'论文模型图_第四页_AB分区精简版.pptx'
with ZipFile(source) as z,ZipFile(dst,'w',ZIP_DEFLATED) as out:
 for item in z.infolist():out.writestr(item.filename,X.tostring(root,encoding='utf-8',xml_declaration=True) if item.filename=='ppt/slides/slide4.xml' else z.read(item.filename))
print(dst)
