"""Restore the archived candidate into a NEW directory; never overwrite an existing checkout."""
import argparse,hashlib,json,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
backup=Path(__file__).resolve().parent;manifest=json.loads((backup/'manifest.json').read_text())
archive=backup/'candidate_full_snapshot.tar.gz'
if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['archive_sha256']:raise ValueError('Archive checksum mismatch')
if args.output.exists():raise FileExistsError('Choose a new directory')
args.output.mkdir(parents=True);destination=args.output.resolve()
old_prefix='home/isee/Catch/catch_it_copy/'
with tarfile.open(archive,'r:gz') as tar:
    for member in tar:
        name=member.name.lstrip('/')
        if name.startswith(old_prefix):name=name[len(old_prefix):]
        target=(destination/name).resolve()
        if not target.is_relative_to(destination):raise ValueError('Unexpected archive member path')
        if not member.isfile():raise ValueError('Unexpected non-file archive entry')
        target.parent.mkdir(parents=True,exist_ok=True)
        stream=tar.extractfile(member)
        with target.open('wb') as output:
            while data:=stream.read(1024*1024):output.write(data)
model=destination/'basket_mobile_bridge/runs/tennis_bridge_seed123/teacher_last.pth'
if hashlib.sha256(model.read_bytes()).hexdigest()!=manifest['model_sha256']:raise ValueError('Restored checkpoint checksum mismatch')
print('Restored candidate:',destination)
print('Model SHA256:',manifest['model_sha256'])
