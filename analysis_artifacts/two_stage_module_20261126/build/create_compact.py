from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET
base=Path('/home/isee/Catch/analysis_artifacts')
src=(base/'paper_overview_20261125/build/create.py').read_text()
exec(src[:src.index("blue='E5EEF8'")])
sh.clear(); nodes.clear();edges.clear()
W,H=1060,570
navy='183744'; blue='77C4DF'; pale='E8F5FB'; orange='F4C6A9'; opale='FFF3EA'; bl='249CC4'; ol='E77936'
# Compact module with restrained typography and shallow fills.
navy='283B48'; blue='B8DEEC'; pale='F2F8FB'; orange='F1D4BF'; opale='FCF6F0'; bl='6DA6BF'; ol='C99D79'
# Legend
shape(20,9,9,9,'',blue,bl,geom='ellipse')
text(34,3,230,22,'跟踪与接球共用',13,navy)
shape(275,9,9,9,'',orange,ol,geom='ellipse')
text(289,3,230,22,'仅接球训练启用',13,navy)
text(20,34,135,24,'观测输入',16,navy,True,'ctr')
text(520,34,135,24,'控制动作',16,navy,True,'ctr')
# Nested policy regions, keeping the original two-stage structure.
shape(240,44,220,260,'',opale,ol,geom='roundRect')
shape(253,52,194,25,'阶段二 · 接球策略',opale,opale,16,navy,True,'rect')
shape(254,157,192,133,'',pale,bl,geom='roundRect')
shape(265,162,170,25,'阶段一 · 跟踪策略',pale,pale,16,navy,True,'rect')
for y,t,c,stroke in [(93,'手部策略  πₕ',orange,ol),(194,'机械臂策略  πₐ',blue,bl),(247,'底盘策略  πᵦ',blue,bl)]:
 shape(275,y,150,32,t,c,stroke,16,navy,False)
for y,t,c,stroke in [(93,'手部状态',opale,ol),(194,'机械臂状态',pale,bl),(247,'底盘状态',pale,bl)]:
 shape(20,y,135,32,t,c,stroke,16,navy)
 edge([(155,y+16),(275,y+16)],False,navy)
for y,t,c,stroke in [(93,'手部动作',opale,ol),(194,'机械臂动作',pale,bl),(247,'底盘动作',pale,bl)]:
 shape(520,y,135,32,t,c,stroke,16,navy)
 edge([(425,y+16),(520,y+16)],False,navy)
shape(20,310,135,29,'目标位置',pale,bl,16,navy)
edge([(155,324),(205,324),(205,210)],False,navy,False)
# Existing state arrows also carry target position after the junction.
# Thin strokes, tighter text margins, and a centered reusable group.
sh[:]=[v.replace('w="25400"','w="11430"').replace('w="12700"','w="9525"').replace('lIns="114300" rIns="114300" tIns="57150" bIns="57150"','lIns="28575" rIns="28575" tIns="0" bIns="0"') for v in sh]
# Reuse a validated minimal OOXML presentation package, replacing its slide.
template=base/'paper_overview_20261125/output/四任务强化学习与视觉蒸馏_论文总架构.pptx'
ns='xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
# Group entire diagram so it can be copied/resized as one module and ungrouped for editing.
group=f'<p:grpSp><p:nvGrpSpPr><p:cNvPr id="1000" name="两阶段强化学习模块（可取消组合）"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="{190*E}" y="{110*E}"/><a:ext cx="{680*E}" cy="{350*E}"/><a:chOff x="0" y="0"/><a:chExt cx="{680*E}" cy="{350*E}"/></a:xfrm></p:grpSpPr>{"".join(sh)}</p:grpSp>'
slide=f'<p:sld {ns}><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>{group}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
import re
out=base/'two_stage_module_20261126/output/两阶段强化学习_精简小模块_v2.pptx'
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
