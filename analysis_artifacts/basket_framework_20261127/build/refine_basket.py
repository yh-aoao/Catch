from pathlib import Path
p=Path('analysis_artifacts/basket_framework_20261127/build/edit_basket.py')
s=p.read_text();prefix=s[:s.index("box(32,16")]
exec(prefix)
# Match the source's compact scientific module and Cambria typography.
box(279,155,149,23,'Observations',15,bold=True)
box(500,155,190,23,'Basket Policy',15,bold=True)
box(864,155,123,23,'Actions',15,bold=True)
# Three training labels only, no prose.
box(271,102,227,38,'① Fixed-base Throwing',14,fill=pale,line=bl,bold=True)
box(515,102,227,38,'② Carrying Curriculum',14,fill=opale,line=ol,bold=True)
box(759,102,228,38,'③ Carry–Throw Bridging',14,fill=opale,line=ol,bold=True)
arrow(498,121,515,121);arrow(742,121,759,121)
# Policy region and shared features.
box(470,187,270,211,'',fill=opale,line=ol)
box(482,194,246,22,'②–③ Mobile-base Throwing',14,bold=True)
box(487,251,66,53,'fθ',23,fill=blue,line=bl,bold=True)
box(479,312,82,24,'Shared',12)
box(579,223,143,57,'',fill=opale,line=ol)
box(584,225,133,19,'②–③ Carrying',13,bold=True)
box(593,246,115,27,'πcarry',20,fill=orange,line=ol,bold=True)
box(579,307,143,77,'',fill=pale,line=bl)
box(583,310,135,22,'① → ②–③',13,bold=True)
box(593,343,115,29,'πthrow',20,fill=blue,line=bl,bold=True)
# Unified state input.
box(279,246,149,62,'oₜ , cₜ',24,fill=pale,line=bl,bold=True)
box(281,312,145,20,'cₜ ∈ {carry, throw}',13)
arrow(428,277,487,277)
arrow(553,266,593,260);arrow(553,291,593,357)
# Selection outside the policy region; both heads output all action components.
box(765,251,69,53,'Select',15,fill=opale,line=ol,bold=True)
arrow(708,260,765,266);arrow(708,357,765,291)
box(864,225,123,31,'Aᵇᵃˢᵉₜ',21,fill=pale,line=bl,bold=True)
box(864,273,123,31,'Aᵃʳᵐₜ',21,fill=pale,line=bl,bold=True)
box(864,321,123,31,'Aʰᵃⁿᵈₜ',21,fill=pale,line=bl,bold=True)
arrow(834,266,864,240);arrow(834,277,864,288);arrow(834,291,864,336)
box(743,208,114,23,'cₜ  (rule)',13,bold=True)
arrow(800,231,800,251,dash=True)
box(479,411,260,24,'oₜ = [srobot, sball, g, aₜ₋₁]',15)
# Center the compact module without altering slide size.
for off in tree.iter(q('a:off')):
 off.set('x',str(int(off.get('x'))-100*9525))
# Add true subscripts to symbolic text while keeping native editable text.
for sp in tree.findall(q('p:sp')):
 tx=sp.find(q('p:txBody'))
 if tx is None:continue
 t=''.join(e.text or '' for e in tx.iter(q('a:t')))
 parts={'fθ':[('f',False),('θ',True)],'πcarry':[('π',False),('carry',True)],'πthrow':[('π',False),('throw',True)],'Aᵇᵃˢᵉₜ':[('A',False),('base, t',True)],'Aᵃʳᵐₜ':[('A',False),('arm, t',True)],'Aʰᵃⁿᵈₜ':[('A',False),('hand, t',True)]}.get(t)
 if parts:
  pp=tx.find(q('a:p'));r=pp.find(q('a:r'));rp=r.find(q('a:rPr'));import copy
  pp.remove(r)
  for text,sub in parts:
   rr=add(pp,'a:r');rrp=copy.deepcopy(rp)
   if sub:rrp.set('baseline','-25000');rrp.set('sz',str(round(int(rrp.get('sz'))*.65)))
   rr.append(rrp);add(rr,'a:t').text=text
cm=add(root,'p:clrMapOvr');add(cm,'a:masterClrMapping')
src=Path('analysis_artifacts/two_stage_module_20261126/output/两阶段强化学习.pptx');dst=Path('analysis_artifacts/basket_framework_20261127/output/两阶段强化学习_Basket精简完善版.pptx')
with zipfile.ZipFile(src) as a,zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED) as b:
 for n in a.infolist():b.writestr(n,E.tostring(root,encoding='utf-8',xml_declaration=True) if n.filename=='ppt/slides/slide2.xml' else a.read(n.filename))
with zipfile.ZipFile(src) as a,zipfile.ZipFile(dst) as b:assert [n for n in a.namelist() if a.read(n)!=b.read(n)]==['ppt/slides/slide2.xml']
print(dst)
