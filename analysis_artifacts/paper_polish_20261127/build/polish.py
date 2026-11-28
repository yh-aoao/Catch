from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from copy import deepcopy
import xml.etree.ElementTree as X
A='http://schemas.openxmlformats.org/drawingml/2006/main';P='http://schemas.openxmlformats.org/presentationml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
ns={'a':A,'p':P};E=9525
for k,v in [('a',A),('p',P),('r',R)]:X.register_namespace(k,v)
base=Path('/home/isee/Catch/analysis_artifacts');source=base/'two_stage_module_20261126/output/论文模型图.pptx';dest=base/'paper_polish_20261127/output/论文模型图_第六页_学术美化版.pptx'
with ZipFile(source) as z:root=X.fromstring(z.read('ppt/slides/slide6.xml'))
tree=root.find('p:cSld/p:spTree',ns)
icons={e.find('p:nvPicPr/p:cNvPr',ns).get('name'):deepcopy(e) for e in tree.findall('p:pic',ns)}
for e in list(tree)[2:]:tree.remove(e)
# Three functional color groups: blue policy, teal forecasting, orange supervision.
C={'ink':'243748','muted':'637381','blue':'EAF1F8','bs':'9AB0C5','teal':'E2F0EC','ts':'7BA99E','accent':'216E61','orange':'FCF3E8','os':'C9A17B','train':'AD7545','neutral':'F5F7F9','gray':'B1BCC6','white':'FFFFFF'}
idx=12000;shapes=[];edges=[];text_bounds=[]
def q(t):return '{'+A+'}'+t
def pp(t):return '{'+P+'}'+t
def run(p,txt,size=1600,bold=False,color=None,baseline=None,math=False):
 r=X.SubElement(p,q('r'));at={'sz':str(size),'lang':'zh-CN','b':'1' if bold else '0'}
 if baseline is not None:at['baseline']=str(baseline)
 pr=X.SubElement(r,q('rPr'),at);f=X.SubElement(pr,q('solidFill'));X.SubElement(f,q('srgbClr'),{'val':color or C['ink']})
 X.SubElement(pr,q('latin'),{'typeface':'Cambria' if math else 'Noto Sans CJK SC'});X.SubElement(pr,q('ea'),{'typeface':'Noto Sans CJK SC'})
 X.SubElement(r,q('t')).text=txt

def para(body,align='ctr'):
 p=X.SubElement(body,q('p'));pr=X.SubElement(p,q('pPr'),{'algn':align});sp=X.SubElement(pr,q('lnSpc'));X.SubElement(sp,q('spcPct'),{'val':'100000'});return p

def sym(p,letter,sub='',sup='',large=False):
 run(p,letter,2200 if large else 1600,True,math=True)
 if sub:run(p,sub,1600,True,baseline=-25000,math=True)
 if sup:run(p,sup,1600,True,baseline=35000,math=True)

def box(x,y,w,h,txt='',fill='white',stroke=None,kind='roundRect',large=False,bold=False,color=None,dash=False,align='ctr'):
 global idx
 idx+=1;sp=X.Element(pp('sp'));nv=X.SubElement(sp,pp('nvSpPr'));X.SubElement(nv,pp('cNvPr'),{'id':str(idx),'name':txt.replace('\n',' ') or 'Region'});X.SubElement(nv,pp('cNvSpPr'));X.SubElement(nv,pp('nvPr'))
 pr=X.SubElement(sp,pp('spPr'));xf=X.SubElement(pr,q('xfrm'));X.SubElement(xf,q('off'),{'x':str(round(x*E)),'y':str(round(y*E))});X.SubElement(xf,q('ext'),{'cx':str(round(w*E)),'cy':str(round(h*E))})
 geom=X.SubElement(pr,q('prstGeom'),{'prst':kind});av=X.SubElement(geom,q('avLst'))
 if kind=='roundRect':X.SubElement(av,q('gd'),{'name':'adj','fmla':'val 6000'})
 if fill is None:X.SubElement(pr,q('noFill'))
 else:f=X.SubElement(pr,q('solidFill'));X.SubElement(f,q('srgbClr'),{'val':C.get(fill,fill)})
 ln=X.SubElement(pr,q('ln'),{'w':'14000'})
 if stroke:
  f=X.SubElement(ln,q('solidFill'));X.SubElement(f,q('srgbClr'),{'val':C.get(stroke,stroke)})
  if dash:X.SubElement(ln,q('prstDash'),{'val':'dash'})
 else:X.SubElement(ln,q('noFill'))
 body=X.SubElement(sp,pp('txBody'));X.SubElement(body,q('bodyPr'),{'anchor':'ctr','wrap':'none','lIns':'28575','rIns':'28575','tIns':'0','bIns':'0'});X.SubElement(body,q('lstStyle'))
 if txt:
  for line in txt.split('\n'):run(para(body,align),line,2200 if large else 1600,bold,color)
 else:para(body,align)
 tree.append(sp);shapes.append((x,y,w,h,txt));return sp

def label(x,y,w,h,t,large=False,bold=False,color=None,align='l'):
 return box(x,y,w,h,t,None,None,'rect',large,bold,color,align=align)
def rewrite(sp,rows):
 body=sp.find('p:txBody',ns)
 for p in list(body.findall('a:p',ns)):body.remove(p)
 for fn in rows:fn(para(body))

def edge(points,train=False,color=None,arrow=True,width=None):
 global idx
 edges.append((points,train))
 for j,((x,y),(xx,yy)) in enumerate(zip(points,points[1:])):
  idx+=1;el=X.Element(pp('cxnSp'));nv=X.SubElement(el,pp('nvCxnSpPr'));X.SubElement(nv,pp('cNvPr'),{'id':str(idx),'name':'Supervision' if train else 'Policy flow'});X.SubElement(nv,pp('cNvCxnSpPr'));X.SubElement(nv,pp('nvPr'))
  pr=X.SubElement(el,pp('spPr'));at={}
  if xx<x:at['flipH']='1'
  if yy<y:at['flipV']='1'
  xf=X.SubElement(pr,q('xfrm'),at);X.SubElement(xf,q('off'),{'x':str(round(min(x,xx)*E)),'y':str(round(min(y,yy)*E))});X.SubElement(xf,q('ext'),{'cx':str(round(abs(xx-x)*E)),'cy':str(round(abs(yy-y)*E))});gg=X.SubElement(pr,q('prstGeom'),{'prst':'line'});X.SubElement(gg,q('avLst'))
  ln=X.SubElement(pr,q('ln'),{'w':str(width or (15000 if train else 20000))});f=X.SubElement(ln,q('solidFill'));X.SubElement(f,q('srgbClr'),{'val':C.get(color or ('train' if train else 'ink'),color or C['ink'])})
  if train:X.SubElement(ln,q('prstDash'),{'val':'dash'})
  if arrow and j==len(points)-2:X.SubElement(ln,q('tailEnd'),{'type':'triangle','w':'sm','len':'sm'})
  tree.append(el)
# Background regions, deliberately different visual weights.
box(30,110,625,1030,fill='neutral')
box(720,110,1245,415,fill='blue')
box(800,575,670,280,fill='teal',stroke='ts',dash=True)
box(1515,575,450,290,fill='orange',stroke='os')
label(40,30,620,50,'(a) 两阶段强化学习',True,True)
label(730,30,1220,50,'(b) 多教师视觉蒸馏',True,True)
# Teacher learning: each task trains its own expert.
for x,w,t in [(45,110,'任务'),(175,150,'阶段一'),(365,150,'阶段二'),(545,100,'教师')]:label(x,145,w,35,t,bold=True,align='ctr')
for i,task in enumerate(['Throw','Roll','Bounce','Basket']):
 y=265+i*190;cy=y+55
 pic=deepcopy(icons[task+'任务示意']);idx+=1;pic.find('p:nvPicPr/p:cNvPr',ns).set('id',str(idx));xf=pic.find('p:spPr/a:xfrm',ns)
 for tag,k,v in [('off','x',50),('off','y',y),('ext','cx',95),('ext','cy',95)]:xf.find('a:'+tag,ns).set(k,str(v*E))
 tree.append(pic);label(40,y+100,115,30,task,bold=True,align='ctr')
 edge([(145,cy),(175,cy)],color='muted')
 if task=='Basket':t1='固定底盘\n投球';t2='移动底盘\n投球'
 else:t1='跟踪拦截';t2='稳定接球'
 box(175,y+20,150,70,t1,'white','bs',bold=True)
 box(365,y+20,150,70,t2,'white','bs',bold=True)
 edge([(325,cy),(365,cy)],color='muted');edge([(515,cy),(550,cy)],color='muted')
 sp=box(550,y+10,95,90,'','white','bs',kind='ellipse')
 rewrite(sp,[lambda p,task=task:sym(p,'π',task.lower(),'T',True)])
 edge([(645,cy),(690,cy)],train=True,arrow=False)
edge([(690,320),(690,1040)],train=True,arrow=False)
label(75,1030,540,40,'独立训练 · 按任务调用',bold=True,align='ctr')
label(75,1080,540,30,'阶段二继承阶段一参数',align='ctr',color=C['muted'])
# Deployable student inputs.
inputs=[('顶部深度','I','t','top'),('腕部深度','I','t','wrist'),('自身状态','s','t',''),('上一动作','a','t−1',''),('任务指令','c','','')]
for j,(title,letter,sub,sup) in enumerate(inputs):
 x=750+j*235
 sp=box(x,155,220,75,'','white','bs',kind='parallelogram')
 rewrite(sp,[lambda p,t=title:run(p,t,1600,True),lambda p,l=letter,s=sub,u=sup:sym(p,l,s,u)])
 if j<2:
  box(x+25,265,170,50,'CNN','white','bs',bold=True)
  edge([(x+110,230),(x+110,265)],color='muted');edge([(x+110,315),(x+110,345)],color='muted')
 else:edge([(x+110,230),(x+110,345)],color='muted')
box(750,345,1160,50,'特征拼接','white','bs',bold=True)
edge([(1340,395),(1340,435)])
sp=box(1200,435,280,70,'','white','bs')
rewrite(sp,[lambda p:run(p,'共享 GRU',2200,True),lambda p:sym(p,'h','t')])
edge([(1480,470),(1540,470),(1540,410),(1440,410),(1440,435)],color='muted',width=15000)
sp=label(1550,415,110,30,'');rewrite(sp,[lambda p:sym(p,'h','t−1')])
edge([(1340,505),(1340,540)])
sp=label(1350,505,100,30,'');rewrite(sp,[lambda p:sym(p,'h','t')])
# Main innovation: predictive features contribute to Catch actions via a residual.
label(825,590,260,35,'Future Stable',True,True,C['accent'])
label(1090,594,350,30,'位移预测与残差融合',bold=True,color=C['accent'])
edge([(1340,540),(815,540),(815,675),(840,675)],color='accent')
sp=box(840,640,285,70,'','white','ts')
rewrite(sp,[lambda p:run(p,'未来位移预测',1600,True,C['accent']),lambda p:(sym(p,'Δp','t+{1,3,5}'),run(p,' ∈ ℝ',1600,math=True),run(p,'9',1600,baseline=35000,math=True))])
edge([(982.5,710),(982.5,740)],color='accent')
box(840,740,285,55,'线性映射  9 → 256','white','ts',bold=True,color=C['accent'])
edge([(1125,767.5),(1200,767.5),(1200,778),(1270,778)],color='accent')
sp=box(1270,748,60,60,'+','white','ts',kind='ellipse',large=True,bold=True,color=C['accent'])
edge([(1340,540),(1485,540),(1485,778),(1330,778)],color='accent')
edge([(1300,808),(1300,875),(1170,875),(1170,900)],color='accent')
# Basket retains a direct shared-state action path, routed outside the auxiliary box.
edge([(1340,540),(1980,540),(1980,880),(1745,880),(1745,900)])
# Training-only position heads share h_t and have different task masks.
label(1535,589,410,36,'位置辅助监督 · 仅训练',bold=True,color=C['train'],align='ctr')
edge([(1980,630),(1640,630)],train=True,arrow=False)
for cx,title,sub,loss,task in [(1640,'物体位置','obj','obj','Catch'),(1850,'篮筐位置','basket','basket','Basket')]:
 edge([(cx,630),(cx,650)],train=True)
 sp=box(cx-90,650,180,65,'','white','os')
 rewrite(sp,[lambda p,t=title:run(p,t,1600,True),lambda p,s=sub:sym(p,'p̂','t',s)])
 edge([(cx,715),(cx,745)],train=True)
 sp=box(cx-90,745,180,45,'','white','os')
 rewrite(sp,[lambda p,l=loss,t=task:(sym(p,'ℒ',l),run(p,' · '+t,1600))])
 sp=label(cx-90,815,180,30,'')
 rewrite(sp,[lambda p,s=sub:(run(p,'真值 ',1600),sym(p,'p','t',s))])
 edge([(cx,815),(cx,790)],train=True)
# Policy heads and explicit command routing.
for x,title,sup in [(1010,'Catch 动作头','C'),(1585,'Basket 动作头','B')]:
 sp=box(x,900,320,80,'','blue','bs')
 rewrite(sp,[lambda p,t=title:run(p,t,2200,True),lambda p,u=sup:sym(p,'u','t',u)])
edge([(1170,980),(1170,1060),(1370,1060)])
edge([(1745,980),(1745,1060),(1550,1060)])
sp=box(1370,1012,180,96,'按 c 选择','blue','bs',kind='diamond',bold=True)
edge([(1460,1108),(1460,1150)])
sp=box(1320,1150,280,65,'','blue','bs')
rewrite(sp,[lambda p:(sym(p,'a','t',large=True),run(p,' = tanh(',1600,math=True),sym(p,'u','t'),run(p,')',1600,math=True))])
# Teacher labels supervise selected pre-tanh logits, not positions or hidden features.
edge([(690,1040),(710,1040)],train=True)
sp=box(710,990,240,100,'','orange','os')
rewrite(sp,[lambda p:run(p,'动作模仿',2200,True),lambda p:(sym(p,'ℒ','imit'),run(p,'(',1600,math=True),sym(p,'u','t'),run(p,', ',1600),sym(p,'a','t','T'),run(p,')',1600,math=True))])
sp=label(700,949,220,32,'');rewrite(sp,[lambda p:sym(p,'a','t','T')])
edge([(1460,1127),(990,1127),(990,1040),(950,1040)],train=True)
sp=label(1100,1090,100,30,'');rewrite(sp,[lambda p:sym(p,'u','t')])
# Quiet, minimal legend. No unverified performance or deployment claims.
edge([(760,1230),(810,1230)],arrow=False,color='muted',width=15000);label(820,1215,235,30,'策略数据流',color=C['muted'])
edge([(1080,1230),(1130,1230)],train=True,arrow=False);label(1140,1215,600,30,'训练监督；真值不作为策略输入',color=C['muted'])
# Guarantee a white canvas and the requested two-font-size policy.
cs=root.find('p:cSld',ns);bg=cs.find('p:bg',ns)
if bg is not None:cs.remove(bg)
bg=X.Element(pp('bg'));bp=X.SubElement(bg,pp('bgPr'));f=X.SubElement(bp,q('solidFill'));X.SubElement(f,q('srgbClr'),{'val':'FFFFFF'});X.SubElement(bp,q('effectLst'));cs.insert(0,bg)
# Connectors go behind nodes and labels, preserving their routed geometry.
children=list(tree)[2:];background=children[:4];connectors=[e for e in children if e.tag==pp('cxnSp')];front=[e for e in children[4:] if e.tag!=pp('cxnSp')]
for e in list(tree)[2:]:tree.remove(e)
for e in background+connectors+front:tree.append(e)
fonts={int(pr.get('sz')) for pr in root.findall('.//a:rPr',ns) if pr.get('sz')}
assert fonts=={1600,2200},fonts
with ZipFile(source) as z,ZipFile(dest,'w',ZIP_DEFLATED) as out:
 for item in z.infolist():out.writestr(item.filename,X.tostring(root,encoding='utf-8',xml_declaration=True) if item.filename=='ppt/slides/slide6.xml' else z.read(item.filename))
with ZipFile(source) as z,ZipFile(dest) as out:
 assert all(z.read(f'ppt/slides/slide{i}.xml')==out.read(f'ppt/slides/slide{i}.xml') for i in range(1,6))
print(dest)
