"""Copy the reviewed standalone mod into its own folder without enabling it.

Refuses to overwrite files edited since this tool's previous install receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import re


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mod-root',type=Path,required=True);parser.add_argument('--game-root',type=Path,required=True);parser.add_argument('--mod-name',default='goldgrizzly');parser.add_argument('--content-root',choices=('addon','mods'),default='addon');args=parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,30}',args.mod_name):raise ValueError('Invalid mod directory name')
    source=args.mod_root.resolve();game=args.game_root.resolve();destination=(game/args.content_root/args.mod_name).resolve()
    if not (source/(args.mod_name+'.ini')).is_file():raise ValueError('Matching mod INI missing')
    if destination.parent!=(game/args.content_root).resolve():raise ValueError('Unexpected destination')
    hashes=json.loads((source.parent/'runtime_sha256.json').read_text());validation=json.loads((source.parent/'validation.json').read_text())
    if not validation['passed']:raise ValueError('Package validation has not passed')
    if validation.get('runtime_manifest_sha256')!=digest(source.parent/'runtime_sha256.json'):raise ValueError('Package changed since validation; run its validator again')
    receipt_path=source.parent/'installation.json';old=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    old_destination=Path(old['destination']).resolve() if old.get('destination') else None
    known_roots={(game/area/args.mod_name).resolve() for area in ('addon','mods')}
    previous=old.get('files',{}) if old_destination in known_roots else {}
    for name,expected in hashes.items():
        if Path(name).name!=name or digest(source/name)!=expected:raise ValueError('Invalid source package entry '+name)
        target=destination/name
        if target.exists() and digest(target) not in (expected,previous.get(name)):raise ValueError('Changed or unrelated destination file: '+str(target))
    destination.mkdir(parents=True,exist_ok=True)
    for name,expected in hashes.items():
        target=destination/name
        if not target.exists() or digest(target)!=expected:shutil.copy2(source/name,target)
        if digest(target)!=expected:raise ValueError('Install verification failed '+name)
    receipt=dict(source=str(source),destination=str(destination),files=hashes,enabled='unchanged',
                 note='Enable '+args.mod_name+' in the Mods menu with deliberate override priority. No base assets or mod-selection state changed.')
    receipt_path.write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))


if __name__=='__main__':main()
