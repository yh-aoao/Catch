from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile,ZIP_DEFLATED
out=Path('/home/isee/Catch/analysis_artifacts/task_icons_20261125')
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

for x,y,label,crop in [(90,35,'Throw · 抛来目标',(0,0,50000,50000)),(540,35,'Roll · 桌面滚动',(50000,0,0,50000)),(90,470,'Bounce · 地面弹跳',(0,50000,50000,0)),(540,470,'Basket · 移动平台持球投框',(50000,50000,0,0))]:
 idx+=1;l,t,r,b=crop
 sh.append(f'<p:pic><p:nvPicPr><p:cNvPr id="{idx}" name="{label}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/><a:srcRect l="{l}" t="{t}" r="{r}" b="{b}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="{x*E}" y="{y*E}"/><a:ext cx="{370*E}" cy="{370*E}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')
 text(x-35,y+370,440,52,label,22,'172B40',True,'ctr')
ns='xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
slide=f'<p:sld {ns}><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>{"".join(sh)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
rels=lambda entries:'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/{t}" Target="{p}"/>' for i,t,p in entries)+'</Relationships>'
files={'ppt/slides/slide1.xml':slide,'_rels/.rels':rels([('rId1','officeDocument','ppt/presentation.xml')]),'ppt/presentation.xml':f'<p:presentation {ns}><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId2"/></p:sldMasterIdLst><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst><p:sldSz cx="{1000*E}" cy="{920*E}"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>','ppt/_rels/presentation.xml.rels':rels([('rId1','slide','slides/slide1.xml'),('rId2','slideMaster','slideMasters/slideMaster1.xml')]),'ppt/slides/_rels/slide1.xml.rels':rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','image','../media/four_task_artwork.png')])}
grp='<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree>'
files['ppt/slideLayouts/slideLayout1.xml']=f'<p:sldLayout {ns} type="blank" preserve="1"><p:cSld name="Blank">{grp}</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
files['ppt/slideLayouts/_rels/slideLayout1.xml.rels']=rels([('rId1','slideMaster','../slideMasters/slideMaster1.xml')])
files['ppt/slideMasters/slideMaster1.xml']=f'<p:sldMaster {ns}><p:cSld>{grp}</p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
files['ppt/slideMasters/_rels/slideMaster1.xml.rels']=rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','image','../media/four_task_artwork.png')])
ct='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
for f,typ in [('ppt/presentation.xml','presentation.main'),('ppt/slides/slide1.xml','slide'),('ppt/slideMasters/slideMaster1.xml','slideMaster'),('ppt/slideLayouts/slideLayout1.xml','slideLayout')]:ct+=f'<Override PartName="/{f}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.{typ}+xml"/>'
files['[Content_Types].xml']=ct+'</Types>'
p=out/'output/四任务小图_Basket移动操作平台.pptx'
with ZipFile(p,'w',ZIP_DEFLATED) as z:
 z.write(out/'build/four_task_artwork.png','ppt/media/four_task_artwork.png')
 for f,v in files.items():z.writestr(f,'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'+v)
print(p)
