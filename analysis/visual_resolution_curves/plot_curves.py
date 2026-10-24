from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
out=Path(__file__).parent
root=Path(r'C:/Users/29491/Downloads')
font=font_manager.FontProperties(fname=r'C:/Windows/Fonts/msyh.ttc').get_name()
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[font,'Arial'],'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False})
series={};rows=[]
for res,old,new in [(112,'catch_dualcam_5000_fixed_10_14','catch112_seed123_to10000'),(160,'catch_dualcam160_5000','catch160_seed123_to10000')]:
 vals={}
 for folder in [root/'distillation_visual/runs'/old,root/new]:
  for f in sorted(folder.glob('eval_*.json')):
   it=int(f.stem.split('_')[1]);d=json.loads(f.read_text())
   assert all(v['complete'] for v in d.values())
   rates=[100*d[t]['successes']/d[t]['episodes'] for t in ['throw','roll','bounce']]
   vals[it]=rates+[sum(rates)/3]
   for t in ['throw','roll','bounce']:
    v=d[t];rows.append([res,it,t,v['successes'],v['episodes'],100*v['successes']/v['episodes']])
 series[res]=vals
with (out/'success_rates.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['resolution','iteration','task','successes','episodes','success_percent']);w.writerows(rows)
fig,axes=plt.subplots(2,2,figsize=(12,8))
colors={112:'#0072B2',160:'#D55E00'}
for ax,(index,title) in zip(axes.flat,[(3,'三任务平均'),(0,'Throw'),(1,'Roll'),(2,'Bounce')]):
 for res in [112,160]:
  vals=series[res];x=sorted(vals);y=[vals[i][index] for i in x]
  ax.plot(x,y,color=colors[res],marker='o' if res==112 else 's',markersize=4,linewidth=1.8,linestyle='-' if res==112 else '--',label=f'{res} × {res}')
  ax.annotate(f'{y[-1]:.1f}%',(x[-1],y[-1]),xytext=(-7,10 if res==160 else -17),textcoords='offset points',ha='right',color=colors[res],fontsize=10)
 ax.axvline(5000,color='#888888',linewidth=1,linestyle=':',zorder=0)
 ax.set(title=title,xlim=(0,10400),ylim=(0,100),xlabel='训练迭代数',ylabel='成功率（%）')
 ax.set_xticks(range(0,10001,2000));ax.set_yticks(range(0,101,20));ax.grid(axis='y',alpha=.2)
fig.suptitle('视觉无标签 Catch 蒸馏：112 × 112 与 160 × 160',fontsize=17,y=.98)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
fig.text(.08,.035,'种子：123；≤5000轮：每任务200回合；续训阶段：每任务300回合。点为实际评估，连线仅引导阅读。',fontsize=9,color='#555555')
fig.text(.08,.012,'竖虚线为续训分界，期间源码/资产发生变化；无第0轮实测值，不补零。不含辅助预测。',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.07,1,.90))
fig.savefig(out/'catch_112_vs_160.png',dpi=300)
fig.savefig(out/'catch_112_vs_160.svg')
fig.savefig(out/'catch_112_vs_160.pdf')
print('points', {k:len(v) for k,v in series.items()})
print('last',{k:v[max(v)] for k,v in series.items()})
