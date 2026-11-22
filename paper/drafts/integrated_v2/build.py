from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path(__file__).parent
D=Document(); sec=D.sections[0]
sec.page_width=Inches(8.5); sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.8); sec.left_margin=sec.right_margin=Inches(.85)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
 s=D.styles[name]; s.font.name='Times New Roman'; s.font.color.rgb=RGBColor(0,0,0)
 s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC')
 s.paragraph_format.space_after=Pt(5)
D.styles['Normal'].font.size=Pt(10.5); D.styles['Normal'].paragraph_format.line_spacing=1.2
D.styles['Normal'].paragraph_format.first_line_indent=Pt(21)
for name,size in [('Title',19),('Heading 1',14),('Heading 2',11.5)]:
 s=D.styles[name]; s.font.size=Pt(size); s.font.bold=True
 s.paragraph_format.first_line_indent=Pt(0); s.paragraph_format.keep_with_next=True
 s.paragraph_format.space_before=Pt(12 if name=='Heading 1' else 8)
D.styles['Caption'].font.size=Pt(9.5); D.styles['Caption'].paragraph_format.first_line_indent=Pt(0)
D.core_properties.author=''; D.core_properties.title='面向动态接取与主动投掷的多技能视觉移动灵巧操作'
D.core_properties.subject='中文论文整合稿'; D.core_properties.keywords='动态接取 移动操作 多教师蒸馏'
def para(text,style=None):
 p=D.add_paragraph(style=style)
 # Mark evidence and unfinished items consistently
 import re
 for part in re.split(r'(【[^】]+】)',text):
  r=p.add_run(part)
  if part.startswith('【'): r.font.color.rgb=RGBColor.from_string('9C2F23')
 return p
from math_helpers import content, node
eq_count=0
def equation(text):
 global eq_count
 eq_count+=1
 p=D.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
 p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.space_before=Pt(4);p.paragraph_format.space_after=Pt(7)
 m=node('oMath',content(eq_count));p._p.append(m)
 r=p.add_run('    ('+str(eq_count)+')');r.font.size=Pt(10)
def table(lines):
 rows=[[x.strip() for x in l.strip('|').split('|')] for l in lines]
 t=D.add_table(rows=0,cols=len(rows[0])); t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 widths=([1.8]+[.99]*(len(rows[0])-1)) if len(rows[0])==6 else ([1.75]+[(6.8-1.75)/(len(rows[0])-1)]*(len(rows[0])-1))
 for i,row in enumerate(rows):
  cells=t.add_row().cells
  trpr=cells[0]._tc.getparent().get_or_add_trPr()
  no=OxmlElement('w:cantSplit');trpr.append(no)
  if i==0: repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
  for j,txt in enumerate(row):
   c=cells[j];c.width=Inches(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   p=c.paragraphs[0];p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=1.05;p.paragraph_format.space_after=Pt(4);p.paragraph_format.space_before=Pt(4)
   p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
   r=p.add_run(txt);r.font.size=Pt(9);r.bold=(i==0)
   cp=c._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
   for edge in ['top','bottom','left','right']:
    e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');b.append(e)
   cp.append(b)
   mar=OxmlElement('w:tcMar')
   for edge in ['top','bottom','left','right']:
    e=OxmlElement('w:'+edge);e.set(qn('w:w'),'75');e.set(qn('w:type'),'dxa');mar.append(e)
   cp.append(mar)
   if i==0: sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E8E8E8');cp.append(sh)
 for row in t.rows[:-1]:
  for c in row.cells:
   for p in c.paragraphs:p.paragraph_format.keep_with_next=True
 D.add_paragraph().paragraph_format.space_after=Pt(0)
lines=(ROOT/'manuscript.txt').read_text().splitlines();i=0
while i<len(lines):
 l=lines[i]
 if l.startswith('|'):
  block=[]
  while i<len(lines) and lines[i].startswith('|'):block.append(lines[i]);i+=1
  table(block);continue
 if l.startswith('# '):
  p=para(l[2:],'Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
 elif l.startswith('### '):para(l[4:],'Heading 2')
 elif l.startswith('## '):para(l[3:],'Heading 1')
 elif l.startswith('@ '):equation(l[2:])
 elif l.startswith('! '):p=para(l[2:],'Caption');p.paragraph_format.keep_with_next=True
 elif l:para(l)
 i+=1
out=ROOT.parent/'多技能视觉移动操作论文_中文整合稿_v2.docx'
for elm in list(D.styles.element.iter(qn('w:pBdr'))):
 elm.getparent().remove(elm)
for elm in list(D.element.iter(qn('w:pBdr'))):
 elm.getparent().remove(elm)
D.save(out)
print(out)
print('paragraphs',len(D.paragraphs),'tables',len(D.tables),'equations',len(D.element.xpath('//m:oMath')))
