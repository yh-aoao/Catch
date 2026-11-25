from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile,ZIP_DEFLATED
out=Path('/home/isee/Catch/analysis_artifacts/overall_framework_20261124')
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

# Native connectors, including dashed training-only supervision.
def connector(points,color='263645',dashed=False,head=True):
 global idx
 for j,((x1,y1),(x2,y2)) in enumerate(zip(points,points[1:])):
  idx+=1
  left,top=min(x1,x2),min(y1,y2);w,h=abs(x2-x1),abs(y2-y1)
  flip=(' flipH="1"' if x2<x1 else '')+(' flipV="1"' if y2<y1 else '')
  dash='<a:prstDash val="dash"/>' if dashed else ''
  end='<a:tailEnd type="triangle" w="med" len="med"/>' if head and j==len(points)-2 else ''
  sh.append(f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{idx}" name="连接线 {idx}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr><p:spPr><a:xfrm{flip}><a:off x="{int(left*E)}" y="{int(top*E)}"/><a:ext cx="{int(w*E)}" cy="{int(h*E)}"/></a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom><a:ln w="25400"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill>{dash}{end}</a:ln></p:spPr></p:cxnSp>')
blue='DDEDFC';green='E4F2CF';orange='FBEAD6'
def node(x,y,w,h,t,fill=blue,size=21):shape(x,y,w,h,t,fill,'4A6070',size,'142B3D',True)
text(240,12,1520,65,'四任务教师强化学习与统一视觉策略蒸馏',34,'142B3D',True,'ctr')
for x,w,t in [(200,205,'阶段一强化学习'),(445,205,'阶段二强化学习'),(680,195,'冻结教师')]:
 shape(x,110,w,44,t,'F0F2F4','F0F2F4',20,'283C4A',True)
# connectors first, behind text boxes
connector([(165,402),(185,402)],head=False)
connector([(185,217),(185,562)],head=False)
for y in [217,332,447,562]:
 connector([(185,y),(215,y)])
 connector([(395,y),(455,y)])
 connector([(625,y),(680,y)])
 connector([(855,y),(890,y)],head=False)
connector([(890,217),(890,562)],head=False)
connector([(890,391),(925,391)])
connector([(390,690),(90,690),(90,450)])
connector([(1055,448),(1055,570)])
connector([(1190,623),(1235,623),(1235,290),(1290,290)])
connector([(1500,210),(1500,255)])
connector([(1500,325),(1500,357)],head=False)
connector([(1370,357),(1630,357)],head=False)
connector([(1370,357),(1370,410)])
connector([(1630,357),(1630,410)])
connector([(1370,490),(1370,525),(1500,525)],head=False)
connector([(1630,490),(1630,525),(1500,525)],head=False)
connector([(1500,525),(1500,565)])
connector([(1725,607),(1815,607)])
# Teacher targets and student predictions meet only at the imitation loss.
connector([(1185,391),(1775,391)],'DD7C22',True)
connector([(1725,590),(1750,590),(1750,442),(1870,442),(1870,430)],'DD7C22',True)
connector([(1700,290),(1740,290),(1740,680),(1590,680)],'DD7C22',True)
node(10,350,155,100,'仿真\n特权状态')
for y,task,stage2 in [(175,'Throw','稳定抓取'),(290,'Roll','稳定抓取'),(405,'Bounce','稳定抓取'),(520,'Basket','移动持球投框')]:
 t=task+'：\n'+('固定底盘投球' if task=='Basket' else '追踪 / 拦截')
 node(215,y,180,84,t,size=20);node(455,y+6,170,72,stage2,size=20);node(680,y+6,175,72,task+' 教师',size=21)
node(120,657,490,65,'任务内随机目标物体与物理参数',size=19)
node(925,335,260,113,'按任务调用教师\n生成完整动作标签',size=21)
node(910,570,280,106,'教师 / 学生混合采集\n任务与物体平衡采样',green,20)
node(1325,100,350,110,'双视角深度图\n机器人状态 + 上一动作\nCatch / Basket 指令',green,20)
node(1290,255,410,70,'共享视觉编码 + 时序记忆',green,22)
node(1270,410,225,80,'Catch：\nFuture Stable 增强',green,20)
node(1530,410,200,80,'Basket：\n动作分支',green,21)
node(1280,565,445,84,'底盘 + 机械臂 + 灵巧手动作',green,21)
node(1815,565,170,84,'机器人\n闭环执行',green,21)
# Orange dashed boxes are separate editable rectangle shapes.
def supervision(x,y,w,h,t):
 global sh
 shape(x,y,w,h,t,orange,'DD7C22',20,'573814',True)
 sh[-1]=sh[-1].replace('<a:ln w="12700">','<a:ln w="19050"><a:prstDash val="dash"/>')
supervision(1775,355,215,75,'动作模仿损失')
supervision(1310,680,400,83,'仅训练辅助监督\n目标位置 / 未来位移 / 篮筐位置')
text(20,755,600,45,'Basket 可先进行持球预训练',17,'52677B')
text(635,775,1340,34,'部署时仅使用学生网络；Basket 新教师接入与完整接后投流程仍需验证',17,'52677B')
ns='xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
slide=f'<p:sld {ns}><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>{"".join(sh)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
rels=lambda entries:'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/{t}" Target="{p}"/>' for i,t,p in entries)+'</Relationships>'
files={'ppt/slides/slide1.xml':slide,'_rels/.rels':rels([('rId1','officeDocument','ppt/presentation.xml')]),'ppt/presentation.xml':f'<p:presentation {ns}><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId2"/></p:sldMasterIdLst><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst><p:sldSz cx="{2000*E}" cy="{820*E}"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>','ppt/_rels/presentation.xml.rels':rels([('rId1','slide','slides/slide1.xml'),('rId2','slideMaster','slideMasters/slideMaster1.xml')]),'ppt/slides/_rels/slide1.xml.rels':rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml')])}
grp='<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree>'
files['ppt/slideLayouts/slideLayout1.xml']=f'<p:sldLayout {ns} type="blank" preserve="1"><p:cSld name="Blank">{grp}</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
files['ppt/slideLayouts/_rels/slideLayout1.xml.rels']=rels([('rId1','slideMaster','../slideMasters/slideMaster1.xml')])
files['ppt/slideMasters/slideMaster1.xml']=f'<p:sldMaster {ns}><p:cSld>{grp}</p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
files['ppt/slideMasters/_rels/slideMaster1.xml.rels']=rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml')])
ct='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
for f,typ in [('ppt/presentation.xml','presentation.main'),('ppt/slides/slide1.xml','slide'),('ppt/slideMasters/slideMaster1.xml','slideMaster'),('ppt/slideLayouts/slideLayout1.xml','slideLayout')]:ct+=f'<Override PartName="/{f}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.{typ}+xml"/>'
files['[Content_Types].xml']=ct+'</Types>'
p=out/'output/整体模型流程图_原生可编辑.pptx'
with ZipFile(p,'w',ZIP_DEFLATED) as z:
 for f,v in files.items():z.writestr(f,'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'+v)
print(p)
