import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
out=Path(__file__).parent
font=font_manager.FontProperties(fname=r'C:/Windows/Fonts/msyh.ttc').get_name()
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[font,'Arial'],'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False})
rows=list(csv.DictReader((out/'success_rates.csv').open(encoding='utf-8-sig')))
data={}
for r in rows:
 if int(r['resolution'])==160:data.setdefault(int(r['iteration']),{})[r['task']]=float(r['success_percent'])
x=sorted(data)
fig,ax=plt.subplots(figsize=(10,5.7))
for task,color,marker,offset in [('throw','#0072B2','o',13),('roll','#009E73','s',-15),('bounce','#D55E00','^',-3),('mean','#333333','D',-16)]:
 y=[sum(data[i].values())/3 if task=='mean' else data[i][task] for i in x]
 ax.plot(x,y,label='三任务均值' if task=='mean' else task.title(),color=color,marker=marker,markersize=4,linewidth=2.3 if task=='mean' else 1.6,linestyle='--' if task=='mean' else '-')
 ax.annotate(f'{y[-1]:.1f}%',(10000,y[-1]),xytext=(7,offset),textcoords='offset points',color=color,fontsize=10)
ax.axvline(5000,color='#999999',linestyle=':',linewidth=1)
ax.set(xlim=(0,11100),ylim=(0,100),xlabel='训练迭代数',ylabel='成功率（%）',title='160 × 160 视觉无标签 Catch 蒸馏成功率（seed123）')
ax.set_xticks(range(0,10001,2000));ax.set_yticks(range(0,101,20));ax.grid(axis='y',alpha=.2)
ax.legend(loc='upper left',ncol=4,frameon=False,fontsize=10)
fig.text(.10,.055,'每任务：≤5000轮测试200回合，续训后测试300回合；点为实测值，未平滑，无第0轮数据。',fontsize=9,color='#555555')
fig.text(.10,.018,'竖虚线标记续训分界；期间源码/资产发生变化。不含辅助预测。',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.09,1,1))
fig.savefig(out/'catch_160_curve.png',dpi=300)
fig.savefig(out/'catch_160_curve.svg')
fig.savefig(out/'catch_160_curve.pdf')
print('5000',data[5000],'10000',data[10000])
