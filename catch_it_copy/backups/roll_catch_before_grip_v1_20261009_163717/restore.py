from pathlib import Path
import hashlib, json
import shutil
folder=Path(__file__).resolve().parent
root=folder.parents[2]
m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
for name in m['restore_scope']:
    if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=m['files'][name]:
        raise SystemExit('Backup checksum mismatch: '+name)
# Preserve current experimental files before restoring the baseline.
from datetime import datetime
saved=folder/('replaced_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
for name in m['restore_scope']:
    dest=root/name
    copy=saved/name; copy.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(dest,copy)
    shutil.copy2(folder/name,dest)
    print('Restored:',name)
print('Replaced files saved:',saved)
