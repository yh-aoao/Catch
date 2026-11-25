from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile,ZIP_DEFLATED
out=Path('/home/isee/Catch/analysis_artifacts/overall_framework_20261124')
E=9525
sh=[];idx=1
def shape(x,y,w,h,text='',fill='FFFFFF',line='CBD5E1',size=21,color='17324D',bold=False,geom='roundRect',align='ctr'):
 global idx
 idx+=1
 ps=''.join(f'<a:p><a:pPr algn="{align}"/><a:r><a:rPr lang="zh-CN" sz="{int(size*100)}" b="{int(bold)}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Noto Sans CJK SC"/><a:ea typeface="Noto Sans CJK SC"/></a:rPr><a:t>{escape(t)}</a:t></a:r><a:endParaRPr lang="zh-CN"/></a:p>' for t in text.split('\n'))
 sh.append(f'<p:sp><p:nvSpPr><p:cNvPr id="{idx}" name="组件 {idx}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{int(x*E)}" y="{int(y*E)}"/><a:ext cx="{int(w*E)}" cy="{int(h*E)}"/></a:xfrm><a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln></p:spPr><p:txBody><a:bodyPr wrap="square" anchor="ctr" lIns="114300" rIns="114300" tIns="57150" bIns="57150"/><a:lstStyle/>{ps}</p:txBody></p:sp>')
def text(x,y,w,h,t,size=20,color='17324D',bold=False,align='l'):
 shape(x,y,w,h,t,'FFFFFF','FFFFFF',size,color,bold,'rect',align)
def arrow(x,y,w=40,h=30):shape(x,y,w,h,'','64829C','64829C',geom='rightArrow')
text(42,25,1500,65,'四任务强化学习教师与视觉学生蒸馏框架',34,bold=True)
text(44,95,1480,38,'先独立训练任务专家，再把四类行为联合蒸馏到一个学生策略',19,'52677B')
# three phase captions
for x,w,n,t,c in [(45,530,'01','两阶段强化学习 · 四个教师','1D5D9B'),(630,390,'02','多任务联合蒸馏','267B68'),(1075,480,'03','统一学生策略与执行','7251A2')]:
 text(x,155,w,45,n+'  '+t,23,c,True)
# shared teacher input
shape(45,220,530,60,'仿真特权状态与任务奖励','EAF3FC','8EB9DD',21,bold=True)
text(55,285,510,32,'各任务独立训练，第二阶段继承第一阶段能力',16,'52677B')
for y,t,d in [(330,'Throw 教师 · 抛来目标接取','阶段一：追踪 / 拦截    阶段二：稳定抓取'),(432,'Roll 教师 · 桌面滚动接取','阶段一：追踪 / 拦截    阶段二：稳定抓取'),(534,'Bounce 教师 · 弹跳目标接取','阶段一：追踪 / 拦截    阶段二：稳定抓取'),(636,'Basket 教师 · 持球投框','阶段一：固定底盘投球    阶段二：移动持球投框')]:
 shape(45,y,530,89,'','F4F8FC','B3CDE3');text(59,y+7,500,34,t,21,'1D5D9B',True);text(59,y+45,500,32,d,16,'52677B')
text(48,742,525,58,'按任务选择目标物体\nCatch 多目标物体，Basket 当前使用网球',17,'52677B')
arrow(582,452,38,34)
# distillation spine
shape(630,220,390,110,'任务与物体分层采样\n调用对应教师生成动作示范','EAF7F1','9BC8B4',21,bold=True)
shape(794,347,55,28,'','7FA99B','7FA99B',geom='downArrow')
shape(630,390,390,123,'教师 / 学生混合采集\n按任务、物体平衡训练\n统一动作接口与控制语义','F2FAF6','B1D5C4',19)
shape(794,530,55,28,'','7FA99B','7FA99B',geom='downArrow')
shape(630,575,390,150,'动作模仿学习\n位置与未来运动辅助监督\nBasket 篮筐位置辅助监督','EAF7F1','9BC8B4',20,bold=True)
text(635,742,380,58,'仿真真值只供教师与训练监督\n学生执行时无需读取目标真值',17,'52677B')
arrow(1027,452,38,34)
# student
shape(1075,220,480,110,'学生输入\n双视角深度图、机器人自身状态\n上一动作、Catch / Basket 任务指令','F3EFFA','C2AFDC',19)
shape(1280,347,55,28,'','A08AB9','A08AB9',geom='downArrow')
shape(1075,390,480,123,'共享视觉编码与时序记忆\nFuture Stable 运动预测增强\nCatch 与 Basket 动作分支','F8F5FC','D4C7E5',21,bold=True)
shape(1280,530,55,28,'','A08AB9','A08AB9',geom='downArrow')
shape(1075,575,480,150,'学生自主输出动作\n底盘、机械臂、灵巧手协同控制\n闭环执行接取或持球投框任务','F3EFFA','C2AFDC',21,bold=True)
text(1080,742,470,58,'部署时仅运行学生网络\n按任务评估成功率与各目标泛化能力',17,'52677B')
text(45,830,1510,45,'框架说明：Basket 持球可作为预训练课程；新教师接入蒸馏及完整“接住后再投框”仍需单独验证',16,'687B8B')
ns='xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
slide=f'<p:sld {ns}><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>{"".join(sh)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
rels=lambda entries:'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/{t}" Target="{p}"/>' for i,t,p in entries)+'</Relationships>'
files={'ppt/slides/slide1.xml':slide,'_rels/.rels':rels([('rId1','officeDocument','ppt/presentation.xml')]),'ppt/presentation.xml':f'<p:presentation {ns}><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId2"/></p:sldMasterIdLst><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst><p:sldSz cx="{1600*E}" cy="{900*E}"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>','ppt/_rels/presentation.xml.rels':rels([('rId1','slide','slides/slide1.xml'),('rId2','slideMaster','slideMasters/slideMaster1.xml')]),'ppt/slides/_rels/slide1.xml.rels':rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml')])}
grp='<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree>'
files['ppt/slideLayouts/slideLayout1.xml']=f'<p:sldLayout {ns} type="blank" preserve="1"><p:cSld name="Blank">{grp}</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
files['ppt/slideLayouts/_rels/slideLayout1.xml.rels']=rels([('rId1','slideMaster','../slideMasters/slideMaster1.xml')])
files['ppt/slideMasters/slideMaster1.xml']=f'<p:sldMaster {ns}><p:cSld>{grp}</p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
files['ppt/slideMasters/_rels/slideMaster1.xml.rels']=rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml')])
ct='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
for f,typ in [('ppt/presentation.xml','presentation.main'),('ppt/slides/slide1.xml','slide'),('ppt/slideMasters/slideMaster1.xml','slideMaster'),('ppt/slideLayouts/slideLayout1.xml','slideLayout')]:ct+=f'<Override PartName="/{f}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.{typ}+xml"/>'
files['[Content_Types].xml']=ct+'</Types>'
p=out/'output/四任务强化学习与蒸馏总框架.pptx'
with ZipFile(p,'w',ZIP_DEFLATED) as z:
 for f,v in files.items():z.writestr(f,'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'+v)
print(p)
