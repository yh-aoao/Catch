from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
R=Path(__file__).parent
im=Image.new('RGB',(1400,1000),'white');d=ImageDraw.Draw(im)
fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
f=ImageFont.truetype(fontpath,40);small=ImageFont.truetype(fontpath,33)
def box(x,y,w,h,txt):
 d.rectangle((x,y,x+w,y+h),fill='#f1f1f1',outline='black',width=3)
 lines=txt.split('\n')
 for i,l in enumerate(lines):
  bb=d.textbbox((0,0),l,font=f);d.text((x+w/2-(bb[2]-bb[0])/2,y+h/2-len(lines)*26+i*52-8),l,fill='black',font=f)
def arrow(a,b,label=None,dash=False):
 if dash:
  import math
  dx,dy=b[0]-a[0],b[1]-a[1];n=int(math.hypot(dx,dy)/20)
  for i in range(0,n,2):d.line((a[0]+dx*i/n,a[1]+dy*i/n,a[0]+dx*min(i+1,n)/n,a[1]+dy*min(i+1,n)/n),fill='black',width=3)
 else:d.line((a,b),fill='black',width=4)
 import math
 ang=math.atan2(b[1]-a[1],b[0]-a[0]);p=[b,(b[0]-18*math.cos(ang-.45),b[1]-18*math.sin(ang-.45)),(b[0]-18*math.cos(ang+.45),b[1]-18*math.sin(ang+.45))];d.polygon(p,fill='black')
 if label:d.text(((a[0]+b[0])/2+6,(a[1]+b[1])/2-40),label,fill='black',font=small)
box(50,35,1300,125,'训练监督：冻结教师动作与仿真物体标签')
box(65,205,450,160,'双视角深度\n本体状态与历史动作\n任务与目标条件')
box(65,405,450,110,'共享 CNN 与 GRU')
box(65,580,450,110,'接取表征 h ＋ Future')
box(65,775,450,110,'Catch 动作头')
box(760,245,520,110,'当前位置 Aux')
box(760,425,520,110,'未来位移 Future')
box(760,610,520,110,'Basket 动作头')
box(760,815,520,110,'任务指令选择与动作适配')
arrow((290,365),(290,405));arrow((290,515),(290,580));arrow((290,690),(290,775))
arrow((515,455),(760,300));arrow((515,460),(760,480));arrow((760,480),(515,635))
arrow((515,480),(760,665));arrow((1020,720),(1020,815));arrow((515,830),(760,870))
arrow((1020,160),(1020,245),'位置标签',True)
arrow((1300,160),(1300,475),None,True);arrow((1300,475),(1280,475),None,True)
d.text((1090,370),'位移标签',fill='black',font=small)
arrow((60,160),(35,830),None,True);arrow((35,830),(65,830),None,True)
d.text((70,920),'教师动作标签监督所选技能的输出',fill='black',font=small)
im.save(R/'architecture.png')
