from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as X
p=Path('/home/isee/Catch/analysis_artifacts/two_stage_module_20261126/output')
a=p/'两阶段强化学习_微型模块_v3.pptx';b=p/'Two_Stage_RL_Compact_English.pptx'
P='http://schemas.openxmlformats.org/presentationml/2006/main'; A='http://schemas.openxmlformats.org/drawingml/2006/main'
ns={'p':P,'a':A}
X.register_namespace('p',P);X.register_namespace('a',A)
labels={'跟踪与接球共用':'Tracking & Catching','仅接球训练启用':'Catching Only','观测输入':'Observations','控制动作':'Actions','阶段二 · 接球策略':'Catching Policy','阶段一 · 跟踪策略':'Tracking Policy'}
symbols={'手部状态':('O','hand','t'),'机械臂状态':('O','arm','t,t−1'),'底盘状态':('O','base','t'),'目标位置':('O','object','t,t−1'),'手部动作':('A','hand','t'),'机械臂动作':('A','arm','t'),'底盘动作':('A','base','t'),'手部策略  πₕ':('π','hand',''),'机械臂策略  πₐ':('π','arm',''),'底盘策略  πᵦ':('π','base','')}
def run(parent,txt,base=False,size=1500):
 r=X.SubElement(parent,'{'+A+'}r');pr=X.SubElement(r,'{'+A+'}rPr',{'lang':'en-US','sz':str(size),'i':'1'})
 if not base:pr.set('baseline','-25000')
 fill=X.SubElement(pr,'{'+A+'}solidFill');X.SubElement(fill,'{'+A+'}srgbClr',{'val':'283B48'})
 X.SubElement(pr,'{'+A+'}latin',{'typeface':'Cambria'});X.SubElement(r,'{'+A+'}t').text=txt
 return pr
with ZipFile(a) as z,ZipFile(b,'w',ZIP_DEFLATED) as out:
 for item in z.infolist():
  data=z.read(item.filename)
  if item.filename=='ppt/slides/slide1.xml':
   root=X.fromstring(data)
   for sp in root.findall('.//p:sp',ns):
    txt=''.join(t.text or '' for t in sp.findall('.//a:t',ns))
    if txt in labels:
     ts=sp.findall('.//a:t',ns);ts[0].text=labels[txt]
     for t in ts[1:]:t.text=''
     for pr in sp.findall('.//a:rPr',ns):
      pr.set('lang','en-US');pr.set('sz','450' if txt in ('跟踪与接球共用','仅接球训练启用') else '550')
      for f in pr.findall('a:latin',ns):f.set('typeface','Cambria')
    if txt in symbols:
     body=sp.find('p:txBody',ns)
     for par in list(body.findall('a:p',ns)):body.remove(par)
     par=X.SubElement(body,'{'+A+'}p');X.SubElement(par,'{'+A+'}pPr',{'algn':'ctr'})
     main,sub,sup=symbols[txt]
     run(par,main,True,750);run(par,sub,False,450)
     if sup:
      pr=run(par,sup,False,400);pr.set('baseline','35000')
   data=X.tostring(root,encoding='utf-8',xml_declaration=True)
  out.writestr(item.filename,data)
print(b)
