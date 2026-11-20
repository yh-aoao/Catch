"""Fetch Open Robotics Coke Can source assets; conversion is local, no code fetched."""
from pathlib import Path
import urllib.request,json,hashlib
BASE='https://raw.githubusercontent.com/osrf/gazebo_models/master/coke_can/'
def fetch():
 root=Path(__file__).resolve().parents[1]/'assets/ycb';dest=root/'coke_can';dest.mkdir(parents=True,exist_ok=True)
 sources={}
 for source,name in [('model.sdf','source_model.sdf'),('model.config','source_model.config'),('meshes/coke_can.dae','source_mesh.dae'),('materials/textures/coke_can.png','texture_map.png')]:
  data=urllib.request.urlopen(BASE+source,timeout=60).read();(dest/name).write_bytes(data)
  sources[name]=dict(url=BASE+source,sha256=hashlib.sha256(data).hexdigest())
 (dest/'sources.json').write_text(json.dumps(sources,indent=2));print('Downloaded source assets',flush=True)
 return root,dest,sources

def convert(root,dest,sources):
 import xml.etree.ElementTree as ET
 import numpy as np
 tree=ET.parse(dest/'source_mesh.dae').getroot()
 for element in tree.iter():element.tag=element.tag.split('}')[-1]
 unit=float(tree.find('./asset/unit').get('meter','1'))
 for node in tree.findall('.//visual_scene//node'):
  assert all(e.tag=='instance_geometry' for e in node),'Unexpected Collada transform'
 meshes=tree.findall('.//geometry/mesh');assert len(meshes)==1
 mesh=meshes[0];arrays={}
 for source in mesh.findall('source'):
  stride=int(source.find('./technique_common/accessor').get('stride','1'))
  arrays[source.get('id')]=np.fromstring(source.find('float_array').text,sep=' ').reshape(-1,stride)
 vertices=mesh.find('vertices');vertex_source=next(e.get('source')[1:] for e in vertices.findall('input') if e.get('semantic')=='POSITION')
 positions=arrays[vertex_source][:,:3]*unit
 out=['# Open Robotics Coke Can; converted millimeters to meters']+['v '+' '.join(map(str,v)) for v in positions]
 uv_source=None
 for triangles in mesh.findall('triangles'):
  mapping={e.get('semantic'):(int(e.get('offset','0')),e.get('source')[1:]) for e in triangles.findall('input')}
  width=max(x[0] for x in mapping.values())+1
  indices=np.fromstring(triangles.find('p').text,sep=' ',dtype=int).reshape(-1,3,width)
  assert len(indices)==int(triangles.get('count'))
  assert mapping['VERTEX'][1]==vertices.get('id')
  if 'TEXCOORD' in mapping:
   uv_id=mapping['TEXCOORD'][1]
   if uv_source is None:
    uv_source=uv_id;out+=['vt '+' '.join(map(str,v[:2])) for v in arrays[uv_id]]
   assert uv_source==uv_id
  for face in indices:
   refs=[]
   for corner in face:
    v=int(corner[mapping['VERTEX'][0]])+1
    refs.append(str(v)+'/'+str(int(corner[mapping['TEXCOORD'][0]])+1) if 'TEXCOORD' in mapping else str(v))
   out.append('f '+' '.join(refs))
 (dest/'textured.obj').write_text('\n'.join(out)+'\n')
 source=ET.parse(dest/'source_model.sdf').getroot();mass=float(source.find('.//inertial/mass').text)
 model=ET.Element('mujoco',model='coke_can');asset=ET.SubElement(model,'asset')
 ET.SubElement(asset,'mesh',name='coke_can_visual_mesh',file='coke_can/textured.obj')
 ET.SubElement(asset,'texture',name='coke_can_texture',type='2d',file='coke_can/texture_map.png')
 ET.SubElement(asset,'material',name='coke_can_material',texture='coke_can_texture')
 body=ET.SubElement(ET.SubElement(model,'worldbody'),'body',name='object')
 inertia=source.find('.//inertial/inertia')
 ET.SubElement(body,'inertial',mass=str(mass),pos='0 0 0',diaginertia=' '.join(inertia.find(k).text for k in ('ixx','iyy','izz')))
 ET.SubElement(body,'geom',type='mesh',mesh='coke_can_visual_mesh',material='coke_can_material',mass='0',contype='0',conaffinity='0')
 ET.SubElement(body,'geom',type='cylinder',size='.033 .06',mass=str(mass),contype='1',conaffinity='1')
 xml=ET.tostring(model,encoding='unicode');(root/'coke_can.xml').write_text(xml)
 (root/'coke_can.provenance.json').write_text(json.dumps(dict(source='Open Robotics gazebo_models/coke_can; NON-YCB',source_files=sources,mass_kg=mass,mass_status='upstream simulation mass; not measured empty can; rigid without liquid slosh',conversion='Collada millimeters to OBJ meters; benchmark fits/recenters cylinder without arbitrary scaling',generated_sha256={name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in ('textured.obj','texture_map.png')},xml_sha256=hashlib.sha256(xml.encode()).hexdigest()),indent=2))
 print('Converted. Upstream mass:',mass,'kg; mesh extents:',np.ptp(positions,axis=0).tolist(),flush=True)

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--offline',action='store_true');a=p.parse_args()
 if a.offline:
  root=Path(__file__).resolve().parents[1]/'assets/ycb';dest=root/'coke_can';sources=json.loads((dest/'sources.json').read_text())
 else:root,dest,sources=fetch()
 convert(root,dest,sources)
if __name__=='__main__':main()
