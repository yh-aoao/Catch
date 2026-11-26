from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET
base=Path('/home/isee/Catch/analysis_artifacts')
src=(base/'paper_overview_20261125/build/create.py').read_text()
exec(src[:src.index("blue='E5EEF8'")])
sh.clear(); nodes.clear();edges.clear()
W,H=1060,570
navy='183744'; blue='77C4DF'; pale='E8F5FB'; orange='F4C6A9'; opale='FFF3EA'; bl='249CC4'; ol='E77936'
# Compact native module, with the original nested tracking/catching organization.
shape(20,15,19,19,'',blue,navy,geom='ellipse')
text(45,10,350,29,'阶段一、二共用的数据流',21,navy)
shape(424,15,19,19,'',orange,navy,geom='ellipse')
text(449,10,330,29,'仅接球训练启用的数据流',21,navy)
text(25,58,230,35,'观测输入',23,navy,True,'ctr')
text(420,58,240,35,'两阶段强化学习',23,navy,True,'ctr')
text(805,58,230,35,'控制动作',23,navy,True,'ctr')
# Outer catching frame and inner tracking frame.
shape(393,105,293,417,'',opale,ol,geom='roundRect')
text(412,111,255,35,'阶段二：接球策略',23,navy,True,'ctr')
shape(410,259,259,247,'',pale,bl,geom='roundRect')
text(424,268,230,34,'阶段一：跟踪策略',23,navy,True,'ctr')
# Three editable network policy blocks.
for x,y,w,h,t,c,stroke in [(435,159,208,72,'π_hand  手部策略',orange,ol),(435,320,208,68,'π_arm  机械臂策略',blue,bl),(435,418,208,68,'π_base  底盘策略',blue,bl)]:
 shape(x,y,w,h,t,c,stroke,23,navy,True)
# Observation rows and action rows; all native text, rounded rectangles and lines.
for y,t,c,stroke in [(160,'手部状态  o_hand',opale,ol),(321,'机械臂状态  o_arm',pale,bl),(419,'底盘状态  o_base',pale,bl)]:
 shape(25,y,230,68,t,c,stroke,22,navy)
 edge([(255,y+34),(435,y+34)],False,navy)
for y,t,c,stroke in [(160,'手部动作  a_hand',opale,ol),(321,'机械臂动作  a_arm',pale,bl),(419,'底盘动作  a_base',pale,bl)]:
 shape(805,y,230,68,t,c,stroke,22,navy)
 edge([(643,y+34),(805,y+34)],False,navy)
# Target observation enters the tracking policies jointly via a clean external bus.
shape(25,510,230,43,'目标位置  o_object',pale,bl,21,navy)
edge([(255,531),(366,531),(366,354)],False,navy,False)
edge([(366,354),(435,354)],False,navy)
edge([(366,452),(435,452)],False,navy)
# The observation arrows overlap last bus segment, intentionally merging at policy input.
# Reuse a validated minimal OOXML presentation package, replacing its slide.
template=base/'paper_overview_20261125/output/四任务强化学习与视觉蒸馏_论文总架构.pptx'
ns='xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
# Group entire diagram so it can be copied/resized as one module and ungrouped for editing.
group=f'<p:grpSp><p:nvGrpSpPr><p:cNvPr id="1000" name="两阶段强化学习模块（可取消组合）"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{W*E}" cy="{H*E}"/><a:chOff x="0" y="0"/><a:chExt cx="{W*E}" cy="{H*E}"/></a:xfrm></p:grpSpPr>{"".join(sh)}</p:grpSp>'
slide=f'<p:sld {ns}><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>{group}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
import re
out=base/'two_stage_module_20261126/output/两阶段强化学习_可编辑模块.pptx'
with ZipFile(template) as z, ZipFile(out,'w',ZIP_DEFLATED) as zz:
 for item in z.infolist():
  data=z.read(item.filename)
  if item.filename=='ppt/slides/slide1.xml': data=slide.encode()
  if item.filename=='ppt/presentation.xml': data=re.sub(rb'<p:sldSz[^>]*/>',f'<p:sldSz cx="{W*E}" cy="{H*E}"/>'.encode(),data)
  if item.filename.endswith('.rels'):
   data=re.sub(rb'<Relationship\b[^>]*Type="[^"]*/image"[^>]*/>',b'',data)
  if item.filename.startswith('ppt/media/'):continue
  zz.writestr(item.filename,data)
with ZipFile(out) as z:
 for n in z.namelist():
  if n.endswith('.xml') or n.endswith('.rels'):ET.fromstring(z.read(n))
print(out)
