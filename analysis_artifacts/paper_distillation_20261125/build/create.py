import xml.etree.ElementTree as X
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile,ZIP_DEFLATED
out=Path('/home/isee/Catch/analysis_artifacts/paper_distillation_20261125')
E=9525
sh=[];idx=1
def shape(x,y,w,h,text='',fill='FFFFFF',line='CBD5E1',size=21,color='17324D',bold=False,geom='roundRect',align='ctr'):
 global idx
 idx+=1
 ps=''.join(f'<a:p><a:pPr algn="{align}"/><a:r><a:rPr lang="zh-CN" sz="{int(size*(.82 if size<30 else 1)*100)}" b="{int(bold)}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Noto Sans CJK SC"/><a:ea typeface="Noto Sans CJK SC"/></a:rPr><a:t>{escape(t)}</a:t></a:r><a:endParaRPr lang="zh-CN"/></a:p>' for t in text.split('\n'))
 sh.append(f'<p:sp><p:nvSpPr><p:cNvPr id="{idx}" name="组件 {idx}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{int(x*E)}" y="{int(y*E)}"/><a:ext cx="{int(w*E)}" cy="{int(h*E)}"/></a:xfrm><a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln></p:spPr><p:txBody><a:bodyPr wrap="square" anchor="ctr" lIns="114300" rIns="114300" tIns="57150" bIns="57150"/><a:lstStyle/>{ps}</p:txBody></p:sp>')
def text(x,y,w,h,t,size=20,color='17324D',bold=False,align='l'):
 shape(x,y,w,h,t,'FFFFFF','FFFFFF',size,color,bold,'rect',align)
def arrow(x,y,w=40,h=30):shape(x,y,w,h,'','64829C','64829C',geom='rightArrow')


nodes=[];edges=[]
def box(x,y,w,h,t,fill='E9EFF7',size=23,stroke='64768B',dashed=False):
 shape(x,y,w,h,t,fill,stroke,size,'172B40',False)
 if dashed:sh[-1]=sh[-1].replace('<a:ln w="12700">','<a:ln w="19050"><a:prstDash val="dash"/>')
 nodes.append((x,y,w,h,t,fill,size,stroke,dashed))
def label(x,y,w,h,t,size=24):
 text(x,y,w,h,t,size,'172B40',True);nodes.append((x,y,w,h,t,'FFFFFF',size,'FFFFFF',False))
def edge(pts,dash=False,color='35495D',head=True):
 global idx
 edges.append((pts,dash,color,head))
 for j,((x,y),(xx,yy)) in enumerate(zip(pts,pts[1:])):
  idx+=1;fl=(' flipH="1"' if xx<x else '')+(' flipV="1"' if yy<y else '')
  d='<a:prstDash val="dash"/>' if dash else ''
  e='<a:tailEnd type="triangle" w="med" len="med"/>' if head and j==len(pts)-2 else ''
  sh.append(f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{idx}" name="Flow {idx}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr><p:spPr><a:xfrm{fl}><a:off x="{int(min(x,xx)*E)}" y="{int(min(y,yy)*E)}"/><a:ext cx="{int(abs(xx-x)*E)}" cy="{int(abs(yy-y)*E)}"/></a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom><a:ln w="25400"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill>{d}{e}</a:ln></p:spPr></p:cxnSp>')
# Panel captions
label(25,12,800,45,'(a) Visual observations and recurrent state')
label(855,12,915,45,'(b) Command-conditioned action prediction')
# input data
box(25,100,235,60,'Top depth image');box(25,190,235,60,'Wrist depth image')
box(300,100,165,60,'CNN');box(300,190,165,60,'CNN')
box(25,290,440,60,'Robot proprioception (36D)')
box(25,375,440,60,'Previous executed action (20D)')
box(25,460,440,60,'Command c: Catch / Basket')
box(515,220,120,230,'Concat.',size=19)
box(685,275,150,115,'GRU\nhₜ (256D)',fill='DDE7F4')
for y in [130,220]:edge([(260,y),(300,y)])
for y in [130,220]:edge([(465,y),(490,y),(490,255 if y==130 else 290),(515,255 if y==130 else 290)])
for y,z in [(320,330),(405,370),(490,410)]:edge([(465,y),(480,y),(480,z),(515,z)])
edge([(635,332),(685,332)])
# Future residual and task heads
box(875,90,285,90,'Future displacement head\n1 / 3 / 5 steps (9D)',fill='E4EFDE',size=22)
box(1205,105,165,60,'Linear\n9D → 256D',fill='E4EFDE',size=21)
box(1240,265,80,65,'+',fill='E4EFDE',size=28)
box(1380,245,175,95,'Catch head\n20D logits',fill='E4EFDE',size=22)
box(1380,410,175,95,'Basket head\n20D logits',fill='EEE6F4',size=22)
box(1600,300,175,125,'Select by c\nand tanh',fill='EEE6F4',size=22)
box(1600,465,175,80,'Action aₜ\n(2 + 6 + 12)',fill='E9EFF7',size=21)
edge([(835,312),(850,312),(850,135),(875,135)])
edge([(1160,135),(1205,135)])
edge([(1370,135),(1395,135),(1395,215),(1280,215),(1280,265)])
edge([(835,332),(1075,332),(1075,297),(1240,297)])
edge([(1320,297),(1380,297)])
edge([(835,355),(855,355),(855,457),(1380,457)])
edge([(1555,292),(1575,292),(1575,330),(1600,330)])
edge([(1555,457),(1575,457),(1575,395),(1600,395)])
edge([(1688,425),(1688,465)])
# compact small annotations
label(930,365,340,40,'Catch: end-effector increments',18)
label(930,492,340,40,'Basket: joint velocity control',18)
# Training strip panel, separated from inference path
label(25,570,950,45,'(c) Training-only supervision')
box(25,650,420,145,'Frozen task teachers\nThrow / Roll / Bounce / Basket\nPrivileged state → action targets',fill='F3F3F3',size=21)
box(515,660,310,125,'Action imitation loss\nSelected logits vs. teacher targets',fill='FFF3E6',size=21,stroke='B47941',dashed=True)
box(890,650,425,145,'Catch auxiliary losses\nCurrent object position\nFuture displacement before contact',fill='FFF3E6',size=21,stroke='B47941',dashed=True)
box(1380,650,395,145,'Basket auxiliary loss\nBasket position in arm-base frame\nPrediction from hₜ',fill='FFF3E6',size=21,stroke='B47941',dashed=True)
edge([(445,722),(515,722)],True,'B47941')
# training supervision endpoints: explicit labels avoid long crossing arrows
label(510,620,320,30,'Teacher and student actions',17)
label(900,620,400,30,'hₜ and future predictions',17)
label(1390,620,380,30,'hₜ',17)
label(25,824,1750,36,'Solid: inference data flow     Dashed boxes: training losses     Auxiliary targets are never policy inputs',18)
# Editable vector SVG and draw.io sources from the same node/edge model.
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="875" viewBox="0 0 1800 875"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#35495D"/></marker></defs><rect width="1800" height="875" fill="white"/>']
mx=X.Element('mxfile',host='app.diagrams.net');di=X.SubElement(mx,'diagram',name='Student distillation');gm=X.SubElement(di,'mxGraphModel',page='0',pageWidth='1800',pageHeight='875');root=X.SubElement(gm,'root');X.SubElement(root,'mxCell',id='0');X.SubElement(root,'mxCell',id='1',parent='0')
for i,(pts,dash,col,head) in enumerate(edges):
 st=' '.join(f'{x},{y}' for x,y in pts);svg.append(f'<polyline points="{st}" fill="none" stroke="#{col}" stroke-width="2"'+(' stroke-dasharray="8 6"' if dash else '')+(' marker-end="url(#arrow)"' if head else '')+'/>')
 c=X.SubElement(root,'mxCell',id='e'+str(i),edge='1',parent='1',style=f'edgeStyle=orthogonalEdgeStyle;rounded=0;strokeWidth=2;strokeColor=#{col};dashed={int(dash)};endArrow={"block" if head else "none"};')
 g=X.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'});X.SubElement(g,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),attrib={'as':'sourcePoint'});X.SubElement(g,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),attrib={'as':'targetPoint'});ar=X.SubElement(g,'Array',attrib={'as':'points'})
 for x,y in pts[1:-1]:X.SubElement(ar,'mxPoint',x=str(x),y=str(y))
for i,(x,y,w,h,t,fill,size,stroke,dash) in enumerate(nodes):
 actual=size*.82*96/72
 svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#{fill}" stroke="#{stroke}" stroke-width="1.5"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
 lines=t.split('\n');dy=actual*1.22
 for j,line in enumerate(lines):svg.append(f'<text x="{x+w/2}" y="{y+h/2+(j-(len(lines)-1)/2)*dy}" dominant-baseline="middle" text-anchor="middle" font-family="Noto Sans CJK SC,Arial,sans-serif" font-size="{actual}" fill="#172B40">{escape(line)}</text>')
 c=X.SubElement(root,'mxCell',id='n'+str(i),value=t,vertex='1',parent='1',style=f'rounded=1;whiteSpace=wrap;html=0;fontFamily=Arial;fontSize={actual};fillColor=#{fill};strokeColor=#{stroke};dashed={int(dash)};')
 X.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
svg.append('</svg>');(out/'output/student_distillation_paper.svg').write_text(''.join(svg))
X.ElementTree(mx).write(out/'output/student_distillation_paper.drawio',encoding='utf-8',xml_declaration=True)
ns='xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
slide=f'<p:sld {ns}><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>{"".join(sh)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
rels=lambda entries:'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/{t}" Target="{p}"/>' for i,t,p in entries)+'</Relationships>'
files={'ppt/slides/slide1.xml':slide,'_rels/.rels':rels([('rId1','officeDocument','ppt/presentation.xml')]),'ppt/presentation.xml':f'<p:presentation {ns}><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId2"/></p:sldMasterIdLst><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst><p:sldSz cx="{1800*E}" cy="{875*E}"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>','ppt/_rels/presentation.xml.rels':rels([('rId1','slide','slides/slide1.xml'),('rId2','slideMaster','slideMasters/slideMaster1.xml')]),'ppt/slides/_rels/slide1.xml.rels':rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml')])}
grp='<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree>'
files['ppt/slideLayouts/slideLayout1.xml']=f'<p:sldLayout {ns} type="blank" preserve="1"><p:cSld name="Blank">{grp}</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
files['ppt/slideLayouts/_rels/slideLayout1.xml.rels']=rels([('rId1','slideMaster','../slideMasters/slideMaster1.xml')])
files['ppt/slideMasters/slideMaster1.xml']=f'<p:sldMaster {ns}><p:cSld>{grp}</p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
files['ppt/slideMasters/_rels/slideMaster1.xml.rels']=rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml')])
ct='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
for f,typ in [('ppt/presentation.xml','presentation.main'),('ppt/slides/slide1.xml','slide'),('ppt/slideMasters/slideMaster1.xml','slideMaster'),('ppt/slideLayouts/slideLayout1.xml','slideLayout')]:ct+=f'<Override PartName="/{f}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.{typ}+xml"/>'
files['[Content_Types].xml']=ct+'</Types>'
p=out/'output/student_distillation_paper.pptx'
with ZipFile(p,'w',ZIP_DEFLATED) as z:
 for f,v in files.items():z.writestr(f,'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'+v)
print(p)
