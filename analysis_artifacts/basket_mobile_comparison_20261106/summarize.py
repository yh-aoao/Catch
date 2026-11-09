from pathlib import Path
import json,math,collections
d=Path(__file__).resolve().parent
summary={};sets={}
def wilson(k,n):
 z=1.96;p=k/n;den=1+z*z/n
 c=(p+z*z/(2*n))/den
 h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [c-h,c+h]
for kind in ['joint','single']:
 summary[kind]={}
 for ball in ['original','tennis_ball','baseball']:
  data=json.loads((d/(kind+'_'+ball+'.json')).read_text());r=data['records']
  assert len(r)==200 and [x['seed'] for x in r]==list(range(20000,20200))
  k=sum(x['success'] for x in r)
  summary[kind][ball]=dict(checkpoint=data['checkpoint'],episodes=len(r),successes=k,
    success_rate=k/len(r),wilson95=wilson(k,len(r)),
    failures=dict(collections.Counter(x['reason'] for x in r if not x['success'])),
    dangerous_collisions=sum(x['collision'] for x in r),
    mean_steps=sum(x['steps'] for x in r)/len(r))
  sets[kind,ball]=r
 summary[kind]['macro_success']=sum(summary[kind][b]['success_rate'] for b in ['original','tennis_ball','baseball'])/3
summary['paired']={}
for ball in ['original','tennis_ball','baseball']:
 a,b=sets['joint',ball],sets['single',ball]
 assert all(x['target']==y['target'] and x['seed']==y['seed'] for x,y in zip(a,b))
 joint_only=sum(x['success'] and not y['success'] for x,y in zip(a,b))
 single_only=sum(y['success'] and not x['success'] for x,y in zip(a,b))
 summary['paired'][ball]=dict(joint_only=joint_only,single_only=single_only,
   single_minus_joint=(single_only-joint_only)/200)
(d/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# 三球联合与单球移动抛球对比（2026-11-06）','',
'每种球、每种模型200回合，共1200回合。确定性动作，纯移动回合，关闭固定位置复习。每种球使用种子20000–20199，联合/单球使用完全相同目标位置；目标x均匀[-0.2,0.2]、y均匀[1.15,1.55]。联合策略保留53维输入，强制按指定球种采样以做分层测试。未训练或更新权重。','',
'| 球种 | 联合37.56 | 对应单球模型 | 单球减联合 |','|---|---:|---:|---:|']
for ball in ['original','tennis_ball','baseball']:
 a,b=summary['joint'][ball],summary['single'][ball]
 lines.append(f"| {ball} | {a['successes']}/200 ({a['success_rate']:.1%}) | {b['successes']}/200 ({b['success_rate']:.1%}) | {(b['success_rate']-a['success_rate'])*100:+.1f}个百分点 |")
lines+=['',f"联合三球宏平均：{summary['joint']['macro_success']:.1%}；单球模型平均：{summary['single']['macro_success']:.1%}。",
'','详细失败原因、95% Wilson区间和配对结果见summary.json；逐回合目标、终止原因和奖励记录见六组JSON。每组200回合仍是对当前场景范围的估计，不能代表实机或范围外表现。']
(d/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
