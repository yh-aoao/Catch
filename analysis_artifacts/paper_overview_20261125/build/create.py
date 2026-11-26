import xml.etree.ElementTree as X
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile,ZIP_DEFLATED
out=Path('/home/isee/Catch/analysis_artifacts/paper_overview_20261125')
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

blue='E5EEF8';green='E4EFDE';purple='EEE7F3';orange='FFF2E3'
label(25,15,650,50,'(a) 四任务教师的两阶段强化学习',26)
label(725,15,390,50,'(b) 多教师联合蒸馏',26)
label(1160,15,815,50,'(c) 统一视觉学生策略',26)
label(205,90,200,38,'第一阶段',21);label(455,90,210,38,'第二阶段',21)
# Four task illustrations are independent cropped picture objects.
def icon(x,y,crop,name):
 global idx
 idx+=1;l,t,r,b=crop
 sh.append(f'<p:pic><p:nvPicPr><p:cNvPr id="{idx}" name="{name}任务示意"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/><a:srcRect l="{l}" t="{t}" r="{r}" b="{b}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="{int(x*E)}" y="{int(y*E)}"/><a:ext cx="{150*E}" cy="{150*E}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')
for y,task,crop,a,b in [(150,'Throw',(0,0,50000,50000),'追踪 / 拦截','稳定抓取'),(340,'Roll',(50000,0,0,50000),'追踪 / 拦截','稳定抓取'),(530,'Bounce',(0,50000,50000,0),'追踪 / 拦截','稳定抓取'),(720,'Basket',(50000,50000,0,0),'固定底盘投球','移动持球投框')]:
 icon(25,y-5,crop,task);label(25,y+140,150,38,task,20)
 box(200,y+30,200,80,a,blue,21)
 box(455,y+30,210,80,b,blue,21)
 edge([(175,y+70),(200,y+70)])
 edge([(400,y+70),(455,y+70)],True,'788696')
 edge([(665,y+70),(695,y+70)],False,'35495D',False)
edge([(695,220),(695,790)],False,'35495D',False)
edge([(695,455),(740,455)])
label(405,135,95,25,'初始化',15)
box(75,965,575,78,'仿真特权状态与任务奖励\n任务内随机目标物体与物理参数','F3F3F3',21)
label(45,890,630,45,'四个教师独立训练，第二阶段继承已有能力',19)
# Distillation, links refer to training stages only.
box(740,180,360,115,'任务与物体平衡采样\n教师 / 学生混合采集','F3F3F3',22)
edge([(920,295),(920,390)])
box(740,390,360,130,'查询对应冻结教师\n转换为完整控制动作标签',blue,22)
edge([(920,520),(920,645)],True,'B47941')
box(740,645,360,115,'动作模仿损失\n教师标签 vs. 学生原始输出',orange,21,'B47941',True)
box(740,850,360,95,'联合优化共享学生参数\n覆盖四任务与多目标条件','F3F3F3',21)
edge([(920,760),(920,850)],True,'B47941')
# Student inputs and shared backbone.
box(1190,100,760,90,'双视角深度图 + 机器人自身状态\n上一执行动作 + Catch / Basket 任务指令',blue,23)
box(1270,250,600,85,'共享视觉编码器 + GRU 时序记忆 hₜ',blue,24)
edge([(1570,190),(1570,250)])
box(1190,440,330,95,'Future Stable\n未来位移预测与残差融合',green,22)
box(1190,610,330,80,'Catch 动作头',green,23)
box(1630,610,320,80,'Basket 动作头',purple,23)
edge([(1430,335),(1430,385),(1355,385),(1355,440)])
edge([(1355,535),(1355,610)])
edge([(1730,335),(1790,335),(1790,610)])
box(1410,760,400,80,'按任务指令选择动作输出',purple,22)
edge([(1355,690),(1355,800),(1410,800)])
edge([(1790,690),(1790,760)])
box(1410,900,400,80,'tanh 与控制尺度转换\n底盘 + 机械臂 + 灵巧手',blue,21)
edge([(1610,840),(1610,900)])
# Explicit training connections, no privileged values enter the policy.
box(1190,1040,760,100,'辅助预测与监督（仅训练）\nCatch：物体位置、未来位移    Basket：篮筐相对位置',orange,21,'B47941',True)
edge([(1270,292),(1155,292),(1155,1090),(1190,1090)],True,'B47941')
edge([(1190,490),(1170,490),(1170,1020),(1410,1020),(1410,1040)],True,'B47941')
edge([(1810,800),(1980,800),(1980,1180),(720,1180),(720,702),(740,702)],True,'B47941')
label(1190,1140,750,30,'辅助损失只用于训练，预测位置真值均不作为策略输入',17)
label(1275,985,510,30,'学生原始输出用于动作模仿监督',17)
label(25,1195,1950,40,'实线：策略数据流    橙色虚线：训练监督    灰色虚线：阶段间权重初始化',19)
edge([(1100,895),(1125,895),(1125,292),(1270,292)],True,'B47941')
edge([(1190,1110),(1138,1110),(1138,920),(1100,920)],True,'B47941')
ns='xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
slide=f'<p:sld {ns}><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>{"".join(sh)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
rels=lambda entries:'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/{t}" Target="{p}"/>' for i,t,p in entries)+'</Relationships>'
files={'ppt/slides/slide1.xml':slide,'_rels/.rels':rels([('rId1','officeDocument','ppt/presentation.xml')]),'ppt/presentation.xml':f'<p:presentation {ns}><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId2"/></p:sldMasterIdLst><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst><p:sldSz cx="{2000*E}" cy="{1250*E}"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>','ppt/_rels/presentation.xml.rels':rels([('rId1','slide','slides/slide1.xml'),('rId2','slideMaster','slideMasters/slideMaster1.xml')]),'ppt/slides/_rels/slide1.xml.rels':rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','image','../media/task_icons.png')])}
grp='<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree>'
files['ppt/slideLayouts/slideLayout1.xml']=f'<p:sldLayout {ns} type="blank" preserve="1"><p:cSld name="Blank">{grp}</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
files['ppt/slideLayouts/_rels/slideLayout1.xml.rels']=rels([('rId1','slideMaster','../slideMasters/slideMaster1.xml')])
files['ppt/slideMasters/slideMaster1.xml']=f'<p:sldMaster {ns}><p:cSld>{grp}</p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
files['ppt/slideMasters/_rels/slideMaster1.xml.rels']=rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','image','../media/task_icons.png')])
ct='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
for f,typ in [('ppt/presentation.xml','presentation.main'),('ppt/slides/slide1.xml','slide'),('ppt/slideMasters/slideMaster1.xml','slideMaster'),('ppt/slideLayouts/slideLayout1.xml','slideLayout')]:ct+=f'<Override PartName="/{f}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.{typ}+xml"/>'
files['[Content_Types].xml']=ct+'</Types>'
p=out/'output/四任务强化学习与视觉蒸馏_论文总架构.pptx'
with ZipFile(p,'w',ZIP_DEFLATED) as z:
 z.write('/home/isee/Catch/analysis_artifacts/task_icons_20261125/build/four_task_artwork_ring.png','ppt/media/task_icons.png')
 for f,v in files.items():z.writestr(f,'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'+v)
print(p)
