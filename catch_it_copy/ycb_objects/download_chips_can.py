"""Download the official YCB Berkeley chips-can scan (no upstream code execution)."""
from pathlib import Path
import urllib.request,tarfile,hashlib,json,tempfile,xml.etree.ElementTree as ET
URL='https://ycb-benchmarks.s3.amazonaws.com/data/berkeley/001_chips_can/001_chips_can_berkeley_meshes.tgz'
def main():
 root=Path(__file__).resolve().parents[1]/'assets/ycb';archive=Path(tempfile.gettempdir())/'catch_ycb_chips_can_official.tgz'
 if not archive.exists():
  with urllib.request.urlopen(URL,timeout=60) as response, archive.with_suffix('.part').open('wb') as target:
   while True:
    data=response.read(1024*1024)
    if not data:break
    target.write(data)
  archive.with_suffix('.part').replace(archive)
 with tarfile.open(archive,'r:gz') as tar:
  members=[m for m in tar.getmembers() if m.isfile()]
  names=[m.name for m in members];print('\n'.join(names),flush=True)
  # Poisson surface is preferred for visuals; fallback to TSDF when unavailable.
  objs=[m for m in members if m.name.endswith('textured.obj')]
  if not objs:raise ValueError('No textured scan in official archive')
  chosen=next((m for m in objs if '/poisson/' in m.name),objs[0]);folder=Path(chosen.name).parent
  textures=[m for m in members if Path(m.name).parent==folder and m.name.lower().endswith('.png')]
  if not textures:raise ValueError('Missing scan texture')
  dest=root/'chips_can';dest.mkdir(exist_ok=True);files={}
  for member,name in [(chosen,'textured.obj'),(textures[0],'texture_map.png')]:
   data=tar.extractfile(member).read();(dest/name).write_bytes(data);files[name]=dict(archive_member=member.name,sha256=hashlib.sha256(data).hexdigest())
 model=ET.Element('mujoco',model='chips_can');asset=ET.SubElement(model,'asset')
 ET.SubElement(asset,'mesh',name='chips_can_visual_mesh',file='chips_can/textured.obj')
 ET.SubElement(asset,'texture',name='chips_can_texture',type='2d',file='chips_can/texture_map.png')
 ET.SubElement(asset,'material',name='chips_can_material',texture='chips_can_texture')
 body=ET.SubElement(ET.SubElement(model,'worldbody'),'body',name='object')
 ET.SubElement(body,'inertial',mass='.205',pos='0 0 0',diaginertia='.00114 .00114 .000144')
 ET.SubElement(body,'geom',type='mesh',mesh='chips_can_visual_mesh',material='chips_can_material',mass='0',contype='0',conaffinity='0')
 # Loader needs a collision geom. Actual benchmark collision is re-fitted later.
 ET.SubElement(body,'geom',type='cylinder',size='.0375 .125',mass='.205',contype='1',conaffinity='1')
 xml=ET.tostring(model,encoding='unicode');(root/'chips_can.xml').write_text(xml)
 files['chips_can.xml']=dict(sha256=hashlib.sha256(xml.encode()).hexdigest())
 (root/'chips_can.provenance.json').write_text(json.dumps(dict(source_url=URL,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),mass_kg=.205,files=files,conversion='Official scan visual; benchmark uses fitted analytical cylinder with homogeneous inertia approximation'),indent=2))
 print('Downloaded official YCB chips_can',flush=True)
if __name__=='__main__':main()
