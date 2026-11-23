"""Read local STL/XML to generate measurable mount dimensions; units mm."""
import json,struct,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
xml=ROOT/'catch_it_copy/assets/urdf/x1_xarm6_leap_right_object.xml'
b=ET.parse(xml).find('.//body[@name="base_link"]')
aa=np.fromstring(b.get('axisangle'),sep=' ');level=Rotation.from_rotvec(aa[:3]/np.linalg.norm(aa[:3])*aa[3])
def mesh(name):
 raw=(ROOT/'catch_it_copy/assets/meshes'/name).read_bytes();n=struct.unpack_from('<I',raw,80)[0]
 d=np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')]);return np.frombuffer(raw,d,offset=84,count=n)['v'].astype(float)*1000

def transform(vertices,pos,quat=None):
 shape=vertices.shape;v=vertices.reshape(-1,3)
 if quat is not None:v=Rotation.from_quat(quat[[1,2,3,0]]).apply(v)
 return level.apply(v+pos).reshape(shape)

def part(name):
 geom=next(x for x in b.findall('geom') if x.get('mesh')==name)
 q=np.fromstring(geom.get('quat','1 0 0 0'),sep=' ');p=np.fromstring(geom.get('pos','0 0 0'),sep=' ')*1000
 return transform(mesh(name+'.STL'),p,q)
chassis=transform(mesh('base_link.STL'),np.zeros(3));lo=chassis.min((0,1));hi=chassis.max((0,1))
arm_xml=np.fromstring(b.find('body[@name="arm_base"]').get('pos'),sep=' ')*1000
arm=level.apply(arm_xml)
plate=transform(mesh('Arm_base_Link.STL'),arm_xml)
parts={name:part(name) for name in ['a4_Link','a5_Link']}
centers={name:(v.min((0,1))+v.max((0,1)))/2 for name,v in parts.items()}
rail_mask=(np.ptp(chassis[:,:,2],axis=1)<.01)&(abs(chassis.mean(1)[:,2]-hi[2])<.01)
rail=chassis[rail_mask];rlo=rail.min((0,1));rhi=rail.max((0,1))
steer=np.array([level.apply(np.fromstring(b.find('body[@name="Suspension%d_Link"]'%i).get('pos'),sep=' ')*1000) for i in range(1,5)])
report=dict(units='mm',source_xml=str(xml),
 interpretation='Forward +Y from steer_fl/fr XML names. Measurements in leveled initial geometry (root +0.5deg applied), not raw base_link coordinates. Chassis bounds exclude separately modeled wheels/accessories.',
 raw_arm_base_position_mm=arm_xml.tolist(),leveled_arm_base_position_mm=arm.tolist(),
 chassis_bounds_mm=[lo.tolist(),hi.tolist()],rail_top_bounds_mm=[rlo.tolist(),rhi.tolist()],
 arm_center_to_front_body_edge_mm=float(hi[1]-arm[1]),arm_center_to_rear_body_edge_mm=float(arm[1]-lo[1]),
 arm_center_to_left_body_edge_mm=float(arm[0]-lo[0]),arm_center_to_right_body_edge_mm=float(hi[0]-arm[0]),
 arm_center_ahead_axle_midpoint_mm=float(arm[1]-steer[:,1].mean()),
 arm_center_behind_front_steer_axis_mm=float(steer[2:,1].mean()-arm[1]),
 arm_center_behind_front_rail_end_mm=float(rhi[1]-arm[1]),
 arm_mount_plane_above_rail_top_mm=float(arm[2]-rhi[2]),
 crossbeams={name:dict(center_mm=c.tolist(),front_body_edge_to_center_mm=float(hi[1]-c[1]),
                      front_rail_end_to_center_mm=float(rhi[1]-c[1]),bounds_mm=[parts[name].min((0,1)).tolist(),parts[name].max((0,1)).tolist()]) for name,c in centers.items()},
 crossbeam_center_spacing_mm=float(centers['a4_Link'][1]-centers['a5_Link'][1]),
 note='Use central circular opening in Arm_base_Link as mount-center marker, not rectangular plate geometric center. Plate local Y extends -73..+63mm. Hardware screw locations and strength are not inferred.')
(OUT/'mount_position_dimensions.json').write_text(json.dumps(report,indent=2))
fig,ax=plt.subplots(figsize=(11,12))
ax.add_collection(PolyCollection(chassis[:,:,:2],facecolors='#ededed',edgecolors='none',alpha=1))
ax.add_collection(PolyCollection(rail[:,:,:2],facecolors='#b8bdc2',edgecolors='none'))
for name,v in parts.items():
 ax.add_collection(PolyCollection(v[:,:,:2],facecolors='#6da8d4',edgecolors='none'))
 c=centers[name];ax.text(170,c[1],name+'\n4545 x 315',va='center',fontsize=10)
ax.add_collection(PolyCollection(plate[:,:,:2],facecolors='#7cb86f',edgecolors='none',alpha=.85))
ax.add_patch(Circle(arm[:2],45,fill=False,edgecolor='#326623',lw=1.5));ax.plot(arm[0],arm[1],'+',color='#bb2222',ms=15,mew=2)
ax.axvline(0,color='gray',linestyle='--',lw=.8)
ax.plot([-185,185],[steer[2:,1].mean()]*2,'--',color='gray',lw=.8)
ax.text(-235,steer[2:,1].mean()+7,'Front steering axis',fontsize=9)
ax.annotate('',xy=(0,hi[1]),xytext=(0,arm[1]),arrowprops=dict(arrowstyle='<->',color='#bb2222',lw=1.8))
ax.text(8,(hi[1]+arm[1])/2,'244.2 mm',color='#bb2222',fontsize=13)
for name,xpos in [('a4_Link',-290),('a5_Link',-330)]:
 c=centers[name];val=hi[1]-c[1]
 ax.plot([xpos,c[0]],[c[1],c[1]],color='#4477aa',lw=.7)
 ax.plot([xpos,-245],[hi[1],hi[1]],color='#4477aa',lw=.7)
 ax.annotate('',xy=(xpos,hi[1]),xytext=(xpos,c[1]),arrowprops=dict(arrowstyle='<->',color='#4477aa'))
 ax.text(xpos-8,(hi[1]+c[1])/2,'%.1f mm'%val,rotation=90,va='center',ha='right',color='#4477aa',fontsize=11)
ax.annotate('Arm mounting center\n(center of circular opening)',xy=arm[:2],xytext=(75,-10),arrowprops=dict(arrowstyle='->'),fontsize=11)
ax.text(0,hi[1]+30,'FRONT (+Y)',ha='center',fontsize=14,weight='bold')
ax.text(0,lo[1]-35,'REAR / CAMERA POSTS',ha='center',fontsize=12)
ax.text(0,-210,'Center to front body edge: 244.2 mm\nCenter to rear body edge: 475.8 mm\nCentered across 500 mm body width\nMounting plane = rail top + 45 + 8 = 53 mm',ha='center',va='center',bbox=dict(facecolor='white',alpha=.95),fontsize=12)
ax.set_title('Mount location from project STL + XML\nTop view; nominal geometry, no wheels or camera frame',fontsize=15)
ax.set_xlim(-375,300);ax.set_ylim(-410,425);ax.set_aspect('equal');ax.set_xlabel('Lateral X (mm)');ax.set_ylabel('Longitudinal Y (mm), leveled model origin')
fig.tight_layout();fig.savefig(OUT/'mount_position_top_view.png',dpi=170);fig.savefig(OUT/'mount_position_top_view.svg')
print(json.dumps(report,indent=2))
