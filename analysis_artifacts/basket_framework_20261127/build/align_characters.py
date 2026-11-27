from pathlib import Path
import zipfile,copy,xml.etree.ElementTree as E
P='http://schemas.openxmlformats.org/presentationml/2006/main';A='http://schemas.openxmlformats.org/drawingml/2006/main';ns={'p':P,'a':A};E.register_namespace('p',P);E.register_namespace('a',A)
q=lambda n:'{'+ns[n.split(':')[0]]+'}'+n.split(':')[1]
folder=Path('analysis_artifacts/basket_framework_20261127/output')
src=folder/'两阶段强化学习_Basket同款样式.pptx'
with zipfile.ZipFile(src) as z:r=E.fromstring(z.read('ppt/slides/slide2.xml'));first=E.fromstring(z.read('ppt/slides/slide1.xml'))
def label(el):return ''.join(t.text or '' for t in el.findall('.//a:t',ns))
refs={label(e):e for e in first.findall('.//p:sp',ns)}
for el in r.findall('.//p:sp',ns):
 text=label(el);reference=None;values=None
 if text.startswith('O') and text[1:] in ('robot','ball','goal','prev'):
  reference='Ohandt';values=['O',text[1:],'t−1' if text=='Oprev' else 't']
 elif text in ('Ahand','Aarm','Abase'):
  reference='Ahandt';values=['A',text[1:],'t']
 elif text in ('πcarry','πthrow','fθ','ct'):
  reference='πhand';values=[text[0],text[1:]]
 elif text=='Observations':reference='Observations';values=[text]
 elif text=='Actions':reference='Actions';values=[text]
 elif 'Mobile Policy' in text:reference='Catching Policy';values=[text]
 elif 'Throwing Policy' in text:reference='Tracking Policy';values=[text]
 elif text.startswith('① Fixed') or text.startswith('② Carrying'):reference='Tracking Only';values=[text]
 elif text.startswith('c:'):reference='Tracking Only';values=[text]
 if reference:
  old=el.find('p:txBody',ns);tx=copy.deepcopy(refs[reference].find('p:txBody',ns));runs=tx.findall('.//a:r/a:t',ns)
  if len(runs)==2 and len(values)==1:values=[text.rsplit(' ',1)[0]+' ',text.rsplit(' ',1)[1]]
  assert len(runs)==len(values),(text,len(runs),values)
  for t,v in zip(runs,values):
   t.text=v
   t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
  if text.startswith('c:'):tx.find('a:p/a:pPr',ns).set('algn','ctr')
  el.remove(old);el.append(tx)
# Verify all mathematical runs retain the exact source formatting, including superscript time.
for el in r.findall('.//p:sp',ns):
 text=label(el)
 if text.startswith('A') and text not in ('Actions',):
  runs=el.findall('p:txBody/a:p/a:r',ns)
  assert len(runs)==3
  assert runs[0].find('a:rPr',ns).get('sz')=='1400'
  assert runs[1].find('a:rPr',ns).get('baseline')=='-25000'
  assert runs[2].find('a:rPr',ns).get('baseline')=='35000'
dst=folder/'两阶段强化学习_Basket字符对齐版.pptx'
with zipfile.ZipFile(src) as a,zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED) as b:
 for n in a.infolist():b.writestr(n,E.tostring(r,encoding='utf-8',xml_declaration=True) if n.filename=='ppt/slides/slide2.xml' else a.read(n.filename))
with zipfile.ZipFile(src) as a,zipfile.ZipFile(dst) as b:assert [n for n in a.namelist() if a.read(n)!=b.read(n)]==['ppt/slides/slide2.xml']
print(dst)
