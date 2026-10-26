"""Verify the Roll baseline; --apply restores only the three integration files."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='Project root; defaults to the original catch_it_copy')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    backup = Path(__file__).resolve().parent
    manifest = json.loads((backup / 'manifest.json').read_text())
    root = args.root.resolve() if args.root else Path(manifest['project_root'])
    for entry in manifest['files']:
        source = backup / 'snapshot' / entry['path']
        if sha(source) != entry['sha256']:
            raise RuntimeError('Snapshot integrity check failed: ' + entry['path'])
    print('Snapshot verified:', len(manifest['files']), 'files')
    for name in manifest['restore_targets']:
        target = root / name
        print(name, 'identical' if target.is_file() and sha(target) == sha(backup/'snapshot'/name) else 'changed')
    if not args.apply:
        print('Check only. Use --apply to restore the three files listed above.')
        return
    saved = backup / ('before_restore_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    for name in manifest['restore_targets']:
        source, target = backup/'snapshot'/name, root/name
        if target.exists():
            keep = saved/name
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, keep)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != sha(source):
            raise RuntimeError('Restore verification failed: ' + name)
    print('Restored. Previous contents saved at', saved)
    print('New standalone modules are left on disk but are no longer selected by the entry point.')


if __name__ == '__main__':
    main()
