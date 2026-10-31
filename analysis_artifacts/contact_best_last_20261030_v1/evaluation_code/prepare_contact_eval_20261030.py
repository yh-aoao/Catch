"""Restore and verify the Contact checkpoint's historical evaluation sources."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import torch


ROOT = Path('/home/isee/Catch')
OUT = ROOT/'analysis_artifacts/contact_multiseed_20261030_v1'
SOURCE = Path('/tmp/catch_contact_eval_sources_20261030/contact')
PREVIOUS = ROOT/'analysis_artifacts/distill_best_multiseed_20261029_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checkpoint = ROOT/'student_best.pth'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['schema'] != 'distillation_aux_contact_v1':
        raise RuntimeError('Expected Contact checkpoint')
    manifest = saved['metadata']['source_asset_sha256']
    old = json.loads((PREVIOUS/'source_audit.json').read_text())['belief']
    extras = old['additional_required_assets']
    expected = dict(manifest)
    expected.update({item['path']: item['sha256'] for item in extras})
    SOURCE.mkdir(parents=True, exist_ok=True)
    origins = {}
    archive = PREVIOUS/'historical_sources_and_assets.tar.gz'
    with tarfile.open(archive, 'r|gz') as tf:
        for member in tf:
            if not member.name.startswith('belief/'):
                continue
            rel = member.name[len('belief/'):]
            if rel not in expected:
                continue
            if not member.isfile() or '..' in Path(rel).parts or Path(rel).is_absolute():
                raise RuntimeError('Unexpected archive member: '+member.name)
            path = SOURCE/rel
            path.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, path.open('wb') as dest:
                shutil.copyfileobj(src, dest)
            if sha(path) != expected[rel]:
                raise RuntimeError('Historical source mismatch: '+rel)
            origins[rel] = 'prior_verified_belief_archive'
    for rel, value in expected.items():
        if rel in origins:
            continue
        current = ROOT/'catch_it_copy'/rel
        if not current.is_file() or sha(current) != value:
            raise RuntimeError('Cannot exactly restore source: '+rel)
        path = SOURCE/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current, path)
        origins[rel] = 'current_file_exact_checkpoint_hash'
    files = []
    for rel, value in manifest.items():
        actual = sha(SOURCE/rel)
        if actual != value:
            raise RuntimeError('Final source audit failed: '+rel)
        files.append(dict(path=rel, expected=value, actual=actual, matching=True, origin=origins[rel]))
    OUT.mkdir(parents=True, exist_ok=True)
    audit = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                 iteration=saved['iteration'], schema=saved['schema'], source_root=str(SOURCE),
                 files=files, exceptions=[], additional_required_assets=extras,
                 restoration_archive=str(archive), restoration_archive_sha256=sha(archive),
                 same_as_prior_contact_checkpoint=sha(checkpoint)==sha(ROOT/'distill_best_ckpt/contact_student_best.pth'))
    (OUT/'source_audit.json').write_text(json.dumps({'contact': audit}, indent=2))
    (OUT/'resolved_checkpoint_config.json').write_text(json.dumps(saved['config'], indent=2))
    print(json.dumps(dict(iteration=saved['iteration'], manifest_files=len(files),
                         all_manifest_hashes_matching=True, extra_assets=len(extras),
                         restored_source=str(SOURCE), output=str(OUT)), indent=2))


if __name__ == '__main__':
    main()
