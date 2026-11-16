from pathlib import Path
import struct,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
R=Path(__file__).resolve().parents[2]; directory=R/'catch_it_copy/assets/meshes'
def load(name):
 raw=(directory/(name+'.STL')).read_bytes();n=struct.unpack_from('<I',raw,80)[0]
 return np.frombuffer(raw,dtype=np.dtype([('normal','<f4',3),('v','<f4',(3,3)),('a','<u2')]),offset=84,count=n)['v'].astype(float)
xml=ET.parse(R/'catch_it_copy/assets/urdf/x1_xarm6_leap_right_object.xml')
body=xml.find('.//body[@name="base_link"]')
parts={}
for name in ['a1_Link','a3_Link','a4_Link','a5_Link','l1_Link','l4_Link','l6_Link','l7_Link']:
 g=next(g for g in body.findall('geom') if g.get('mesh')==name)
 q=np.fromstring(g.get('quat'),sep=' ');rot=Rotation.from_quat(q[[1,2,3,0]])
 parts[name]=rot.apply(load(name).reshape(-1,3)).reshape(-1,3,3)+np.fromstring(g.get('pos'),sep=' ')
p=body.find('body[@name="arm_base"]');parts['Arm_base_Link']=load('Arm_base_Link')+np.fromstring(p.get('pos'),sep=' ')
fig=plt.figure(figsize=(14,7))
for idx,names in enumerate([['a4_Link','a5_Link','l6_Link','l7_Link','Arm_base_Link'],['a1_Link','a3_Link','l1_Link','l4_Link']]):
 ax=fig.add_subplot(1,2,idx+1,projection='3d')
 allv=[]
 for name in names:
  v=parts[name]*1000
  if idx==1:v=np.minimum(v, np.array([10000,10000,440]))
  allv.append(v.reshape(-1,3))
  color='#e78944' if name.startswith('l') else ('#91bfe2' if name.startswith('a') else '#a2bd86')
  ax.add_collection3d(Poly3DCollection(v,facecolors=color,edgecolors='#444444',linewidths=.10,alpha=1))
  c=v.mean((0,1));ax.text(c[0],c[1],v[:,:,2].max()+12,name,fontsize=10)
 vv=np.concatenate(allv);lo=vv.min(0);hi=vv.max(0);mid=(lo+hi)/2;rad=max(hi-lo)/2*1.2
 ax.set_xlim(mid[0]-rad,mid[0]+rad);ax.set_ylim(mid[1]-rad,mid[1]+rad);ax.set_zlim(mid[2]-rad,mid[2]+rad);ax.set_box_aspect((1,1,1));ax.view_init(25,-55)
 ax.set_xlabel('Base X (mm)');ax.set_ylabel('Base Y (mm)');ax.set_zlabel('Base Z (mm)')
 ax.set_title(['Arm plate + two 90 x 45 x 45 brackets','Camera post feet: two 45 x 45 x 45 brackets\n(posts shortened for display)'][idx])
fig.tight_layout();fig.savefig(Path(__file__).with_name('bracket_locations.png'),dpi=160)
