from pathlib import Path
from copy import deepcopy
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from math_helpers import parse,node
import re,json,zipfile,hashlib
R=Path(__file__).parent
D=Document('/tmp/paper_word_v2/rss_template/paper_template.docx')
sect=deepcopy(D.sections[-1]._sectPr)
for el in list(D.element.body):D.element.body.remove(el)
D.element.body.append(sect)
cols=OxmlElement('w:cols');cols.set(qn('w:num'),'1');cols.set(qn('w:space'),'360');D.sections[0]._sectPr.append(cols)
for n in ['Title','Heading 1','Heading 2','Caption','Reference']:
 if n not in D.styles:D.styles.add_style(n,WD_STYLE_TYPE.PARAGRAPH)
for n in ['Normal','Title','Heading 1','Heading 2','Caption','Reference']:
 s=D.styles[n];s.font.name='Times New Roman';s.font.size=Pt(11);s.font.bold=False;s.font.italic=False;s.font.color.rgb=RGBColor(0,0,0)
 s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC')
 pf=s.paragraph_format;pf.line_spacing=Pt(15);pf.space_before=Pt(0);pf.space_after=Pt(3);pf.first_line_indent=Pt(11);pf.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
 for el in list(s._element.iter(qn('w:numPr'))):el.getparent().remove(el)
D.styles['Title'].font.size=Pt(20);D.styles['Title'].paragraph_format.line_spacing=Pt(26);D.styles['Title'].paragraph_format.space_after=Pt(9)
for n in ['Heading 1','Heading 2']:
 s=D.styles[n];s.font.bold=True;s.paragraph_format.first_line_indent=Pt(0);s.paragraph_format.keep_with_next=True;s.paragraph_format.space_before=Pt(6);s.paragraph_format.space_after=Pt(3)
D.styles['Heading 1'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
D.styles['Caption'].font.size=Pt(9);D.styles['Caption'].paragraph_format.first_line_indent=Pt(0)
D.styles['Reference'].font.size=Pt(9);D.styles['Reference'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.LEFT;D.styles['Reference'].paragraph_format.line_spacing=Pt(11);D.styles['Reference'].paragraph_format.left_indent=Pt(12);D.styles['Reference'].paragraph_format.first_line_indent=Pt(-12);D.styles['Reference'].paragraph_format.space_after=Pt(5)
D.core_properties.author='';D.core_properties.last_modified_by='';D.core_properties.title='预测辅助的多技能视觉移动灵巧接取与投掷';D.core_properties.subject='';D.core_properties.comments='';D.core_properties.keywords=''
for sec in D.sections:
 for hf in [sec.header,sec.footer]:
  for p in hf.paragraphs:p.clear()
def para(txt,style='Normal'):
 p=D.add_paragraph(style=style)
 for part in re.split(r'(\*\*.*?\*\*|【[^】]+】)',txt):
  bold=part.startswith('**');r=p.add_run(part[2:-2] if bold else part);r.bold=bold
  if part.startswith('【'):r.font.color.rgb=RGBColor.from_string('9C2F23')
 return p
count=0
def eq(txt):
 global count
 count+=1
 # Semantic line breaks keep math within one column.
 if txt.startswith('h_t = GRU'):seq=['v_t = [E_1(D_t^1),E_2(D_t^2)],','h_t = GRU_θ([v_t,p_t,a_{t−1},e(c_t)],h_{t−1}).']
 elif txt.startswith('L_{future}'):seq=['L_{future}^τ = (1/(|τ| |H|))',' · Σ_{t∈τ,j∈H} m_t^{(j)}ρ(Δ!x_t^{(j)}−Δx_t^{(j)}).']
 elif txt.startswith('L = Σ'):seq=['L = Σ_{k∈K}w_k[L_{imit}^k',' + I_c(k)(0.1L_{pos}^k+0.1L_{future}^k)].']
 elif txt.startswith('Δ!X_t'):seq=['Δ!X_t = P_{future}(h_t),','~h_t = h_t + W_f Δ!X_t + b_f.']
 elif txt.startswith('!q_t'):seq=['!q_t = P_{pos}(h_t),','L_{pos} = ρ(!q_t/s_q − q_t/s_q).']
 else:seq=[txt]
 p=D.add_paragraph();p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.space_after=Pt(4);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
 for i,line in enumerate(seq):
  if i:p.add_run().add_break()
  m=node('oMath',parse(line));p._p.append(m)
 r=p.add_run(' ('+str(count)+')');r.font.size=Pt(9)
def table(ls):
 rows=[[x.strip() for x in l.strip('|').split('|')] for l in ls]
 t=D.add_table(rows=0,cols=4);t.autofit=False
 for j,w in enumerate([1.18,.7,.7,.7]):t.columns[j].width=Inches(w)
 tw=t._tbl.tblPr.find(qn('w:tblW'));tw.set(qn('w:w'),'4723');tw.set(qn('w:type'),'dxa')
 widths=[1.18,.7,.7,.7]
 for i,row in enumerate(rows):
  cells=t.add_row().cells
  pr=cells[0]._tc.getparent().get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
  if i==0:pr.append(OxmlElement('w:tblHeader'))
  for j,txt in enumerate(row):
   c=cells[j];c.width=Inches(widths[j]);p=c.paragraphs[0];p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2);p.paragraph_format.line_spacing=Pt(11);p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
   r=p.add_run(txt);r.font.size=Pt(8);r.bold=i==0
   cp=c._tc.get_or_add_tcPr();bo=OxmlElement('w:tcBorders')
   for edge in ['top','bottom','left','right']:
    x=OxmlElement('w:'+edge);x.set(qn('w:val'),'single');x.set(qn('w:sz'),'4');x.set(qn('w:color'),'D9D9D9');bo.append(x)
   cp.append(bo)
   if i==0:sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'ECECEC');cp.append(sh)
 for row in t.rows[:-1]:
  for c in row.cells:
   for p in c.paragraphs:p.paragraph_format.keep_with_next=True
lines=(R/'manuscript.txt').read_text().splitlines();i=0;refs=False
while i<len(lines):
 l=lines[i]
 if l.startswith('# '):
  p=para(l[2:],'Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
  sec=D.add_section(WD_SECTION_START.CONTINUOUS);sec._sectPr.find(qn('w:cols')).set(qn('w:num'),'2')
 elif l=='@@REFERENCES':
  sec=D.add_section(WD_SECTION_START.NEW_PAGE);sec._sectPr.find(qn('w:cols')).set(qn('w:num'),'2');refs=True
 elif l.startswith('### '):
  if l=='### A 两阶段专用教师':
   p=D.add_paragraph();p.paragraph_format.first_line_indent=Pt(0);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
   p.paragraph_format.line_spacing=1.0
   p.paragraph_format.keep_with_next=True
   p.add_run().add_picture(str(R/'architecture.png'),width=Inches(3.08))
   p=para('图1 多教师视觉蒸馏与学生推理数据流。虚线仅表示训练监督；当前位置预测不进入动作头，Future 特征进入接取分支。','Caption')
  para(l[4:],'Heading 2')
 elif l.startswith('## '):para(l[3:],'Heading 1')
 elif l.startswith('@ '):eq(l[2:])
 elif l.startswith('! '):p=para(l[2:],'Caption');p.paragraph_format.keep_with_next=True
 elif l.startswith('|'):
  block=[]
  while i<len(lines) and lines[i].startswith('|'):block.append(lines[i]);i+=1
  table(block);continue
 elif l:para(l,'Reference' if refs else 'Normal')
 i+=1
for e in list(D.styles.element.iter(qn('w:pBdr'))):e.getparent().remove(e)
out=R/'RSS2027首轮论文_中文稿.docx';D.save(out)
# Remove obsolete example footnote contents inherited from official template.
from lxml import etree
with zipfile.ZipFile(out) as z:parts={n:z.read(n) for n in z.namelist()}
if 'word/footnotes.xml' in parts:
 root=etree.fromstring(parts['word/footnotes.xml'])
 for e in list(root):
  if e.get(qn('w:id')) not in ['-1','0']:root.remove(e)
 parts['word/footnotes.xml']=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in parts.items():z.writestr(n,b)
print(out);print('equation_blocks',count)
