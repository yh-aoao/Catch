"""Verify immutable local snapshot; optionally restore into a NEW directory."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--restore-to',type=Path);a=p.parse_args()
base=Path(__file__).resolve().parent
m=json.loads((base/'MANIFEST.json').read_text())
for name,entry in m['files'].items():
 f=base/name
 if not f.is_file() or f.stat().st_size!=entry['bytes'] or hashlib.sha256(f.read_bytes()).hexdigest()!=entry['sha256']:
  raise SystemExit('FAILED: '+name)
print('Verified',len(m['files']),'snapshot files')
if a.restore_to:
 dest=a.restore_to.resolve()
 if dest.exists():raise SystemExit('Refusing to overwrite existing directory: '+str(dest))
 shutil.copytree(base/'snapshot/project',dest)
 shutil.copytree(base/'snapshot/models',dest/'baseline_models')
 shutil.copytree(base/'snapshot/evaluation',dest/'baseline_evaluation')
 print('Restored to',dest)
