"""Download selected elpis-lab YCB assets; no repository code is executed."""
import argparse, json, hashlib, urllib.request
from pathlib import Path
import xml.etree.ElementTree as ET

def main():
    p=argparse.ArgumentParser(); p.add_argument('--objects', nargs='+', default=['tennis_ball','baseball','tomato_soup_can']); a=p.parse_args()
    revision=json.load(urllib.request.urlopen('https://api.github.com/repos/elpis-lab/YCB_Dataset/commits/main'))['sha']
    root=Path(__file__).resolve().parents[1]/'assets/ycb'; root.mkdir(parents=True,exist_ok=True)
    base='https://raw.githubusercontent.com/elpis-lab/YCB_Dataset/'+revision+'/'
    for name in a.objects:
        if not name.replace('_','').isalnum(): raise ValueError('Invalid object name')
        xml=urllib.request.urlopen(base+'ycb/'+name+'.xml').read(); tree=ET.fromstring(xml)
        files={name+'.xml':xml}
        for item in tree.findall('./asset/*'):
            f=item.get('file')
            if f:
                if Path(f).is_absolute() or '..' in Path(f).parts: raise ValueError('Unsafe path')
                files[f]=urllib.request.urlopen(base+'ycb/'+f).read()
        for f,data in files.items():
            dest=root/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        (root/(name+'.provenance.json')).write_text(json.dumps(dict(repository=base,revision=revision,files={f:hashlib.sha256(d).hexdigest() for f,d in files.items()}),indent=2))
        print('Downloaded', name, flush=True)
    (root/'UPSTREAM_LICENSE').write_bytes(urllib.request.urlopen(base+'LICENSE').read())
if __name__=='__main__':main()
