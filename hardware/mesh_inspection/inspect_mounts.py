from pathlib import Path
import struct,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT=Path(__file__).resolve().parents[2]
DIR=ROOT/'catch_it_copy/assets/meshes'
def load(name):
 raw=(DIR/name).read_bytes(); n=struct.unpack_from('<I',raw,80)[0]
 dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
 if len(raw)!=84+50*n: raise ValueError(name)
 return np.frombuffer(raw, dtype=dtype,offset=84,count=n)['vertices'].astype(float)*1000
names=['a1_Link.STL','a3_Link.STL','a4_Link.STL','a5_Link.STL','l1_Link.STL','l4_Link.STL','l6_Link.STL','l7_Link.STL','Arm_base_Link.STL']
summary={}
fig=plt.figure(figsize=(15,14))
for i,name in enumerate(names):
 v=load(name);lo=v.min((0,1));hi=v.max((0,1));ext=hi-lo
 summary[name]={'bounds_mm':[lo.tolist(),hi.tolist()],'dimensions_mm':ext.tolist(),'triangles':len(v)}
 ax=fig.add_subplot(3,3,i+1,projection='3d')
 ax.add_collection3d(Poly3DCollection(v,facecolors='#91bfe2',edgecolors='#34516b',linewidths=.08,alpha=.85))
 center=(lo+hi)/2;r=ext.max()/2*1.1
 ax.set_xlim(center[0]-r,center[0]+r);ax.set_ylim(center[1]-r,center[1]+r);ax.set_zlim(center[2]-r,center[2]+r)
 ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-55)
 ax.set_title(name+'\n'+' x '.join('%.1f'%x for x in ext)+' mm',fontsize=11)
 ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
fig.tight_layout();fig.savefig(Path(__file__).with_name('mount_candidates.png'),dpi=150)
Path(__file__).with_name('dimensions.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
