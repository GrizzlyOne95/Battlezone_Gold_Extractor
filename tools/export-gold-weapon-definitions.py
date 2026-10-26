"""Recover weapon presentation records and a conservative BZR recreation map.

Native gameplay numeric fields are not inferred from adjacent untyped words.
Optional executable schema inspection needs pefile and capstone.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import sys

from asura_scene import name_hash, read_animation


FAMILIES={
    'Main_Gun':('cannon','Direct-fire cannon; native balance and event timing unresolved'),
    'Pistol_Weapon':('cannon','Single-shot cannon profile; native balance unresolved'),
    'Autocannon':('machinegun','Best first approximation: stock machinegun ballistics plus Gold audio/model'),
    'Gatling_Gun':('machinegun','Looping rapid fire; spin-up/heat needs an authored controller'),
    'Railgun':('snipergun','Precision-shot approximation; charge and penetration need separate support'),
    'Artillery_weapon':('mortar','Ballistic artillery; native arc, blast and range data unresolved'),
    'Unguided_rocket_launcher':('rocket','Unguided projectile/launcher; chamber/reload state needs controller'),
    'Guided_Missile_Launcher':('missile','Homing approximation; source lock-on and separate rings need controller'),
    'Multi_missile_launcher':('missile','Multiple-launch approximation; burst/targeting state needs controller'),
    'Laser_missile_launcher':('missile','Guidance approximation only; laser steering requires target support'),
    'PlayerTank_Weapon_Cannon_80s':('cannon','Classic visual cannon variant; gameplay balance unresolved'),
}


def controller_record(data,models,clips):
    if len(data)<20:return None
    version,kind,flags,model,clip=struct.unpack_from('<5I',data)
    if version!=1 or kind!=1004 or model not in models or clip not in clips:return None
    return dict(model=models[model],clip=clips[clip]['name'],clip_duration=clips[clip]['duration'],
                model_hash=f'{model:08x}',clip_hash=f'{clip:08x}',record_kind=kind,header_flags=flags,
                byte_length=len(data),unknown_fields='All bytes preserved; flags and numeric tails are not weapon balance fields')


def executable_schema(path):
    import pefile
    import capstone
    data=path.read_bytes();pe=pefile.PE(data=data);base=pe.OPTIONAL_HEADER.ImageBase
    md=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);md.detail=True
    def read(va,size):return data[pe.get_offset_from_rva(va-base):pe.get_offset_from_rva(va-base)+size]
    def string(va):
        try:return read(va,200).split(b'\0',1)[0].decode('ascii')
        except Exception:return None
    # Traced call site: derived projectile-weapon property reader, GOG-style static VA.
    start=0x140461e00;instructions=[];properties=[];current=None;defaults={};default_float=None;default_int=None
    for ins in md.disasm(read(start,4096),start):
        instructions.append(f'{ins.address:016x}: {ins.mnemonic} {ins.op_str}')
        ops=ins.operands
        if ins.mnemonic=='movss' and len(ops)==2 and ops[1].type==capstone.x86.X86_OP_MEM and ops[1].mem.base==capstone.x86.X86_REG_RIP:
            value=struct.unpack('<f',read(ins.address+ins.size+ops[1].mem.disp,4))[0];defaults[ins.reg_name(ops[0].reg)]=value
        if ins.mnemonic=='movaps' and ins.op_str.startswith('xmm3,'):default_float=defaults.get(ins.reg_name(ops[1].reg))
        if ins.mnemonic=='xorps' and ins.op_str=='xmm3, xmm3':default_float=0.
        if ins.mnemonic=='xor' and ins.op_str=='r9d, r9d':default_int=0
        if ins.mnemonic=='mov' and len(ops)==2 and ops[0].type==capstone.x86.X86_OP_REG and ins.reg_name(ops[0].reg)=='r9d' and ops[1].type==capstone.x86.X86_OP_IMM:default_int=ops[1].imm
        if ins.mnemonic=='lea' and len(ops)==2 and ops[0].reg==capstone.x86.X86_REG_R8 and ops[1].type==capstone.x86.X86_OP_MEM and ops[1].mem.base==capstone.x86.X86_REG_RIP:
            name=string(ins.address+ins.size+ops[1].mem.disp)
            if name and name.startswith('.'):current=dict(path=name,string_reference_va=hex(ins.address))
        if current and ins.mnemonic=='call' and len(ops)==1 and ops[0].type==capstone.x86.X86_OP_MEM and ops[0].mem.base==capstone.x86.X86_REG_RAX:
            slot=ops[0].mem.disp;current.update(reader_slot=hex(slot),call_va=hex(ins.address),
                value_type={0x20:'float',0x50:'bool',0x60:'integer',0x80:'identifier_or_reference'}.get(slot,'unverified'),
                engine_fallback=default_float if slot==0x20 else default_int if slot in (0x50,0x60) else None,
                confidence='Static reader schema/fallback only; not a weapon instance definition')
            properties.append(current);current=None
        if ins.mnemonic=='ret':break
    paths={p['path'] for p in properties}
    if '.RateOfFire' not in paths or '.BulletsPerShot' not in paths:raise ValueError('Executable does not match traced weapon reader; refusing schema claims')
    return dict(executable=str(path),sha256=hashlib.sha256(data).hexdigest(),reader_va=hex(start),properties=properties),instructions


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dump-root',type=Path,required=True);parser.add_argument('--executable',type=Path);args=parser.parse_args()
    base=args.dump_root.resolve();out=base/'weapon_definitions';out.mkdir(exist_ok=True)
    models={name_hash(name):name for name in FAMILIES};clips={}
    for path in (base/'raw/chars__actors_pc/HCAN_chunk').glob('*.dat'):
        a=read_animation(path.read_bytes());clips[name_hash(a.name)]=dict(name=a.name,duration=a.duration,source=str(path))
    records=[];blueprints=[]
    for path in (base/'raw/chars__actors_pc/CONA_chunk').glob('*.dat'):
        data=path.read_bytes();record=controller_record(data,models,clips)
        if record:
            dest=out/'native_controllers'/path.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
            record.update(source=str(path),copy=dest.relative_to(out).as_posix(),sha256=hashlib.sha256(data).hexdigest());records.append(record)
    for path in (base/'raw/misc__common_asr/BLUE_chunk').glob('*.dat'):
        data=path.read_bytes();refs=[]
        for key,name in models.items():
            cursor=0
            while True:
                offset=data.find(struct.pack('<I',key),cursor)
                if offset<0:break
                cursor=offset+4
                if offset>=12:
                    field,kind,node=struct.unpack_from('<3I',data,offset-12)
                    if kind==12 and field==node:
                        refs.append(dict(model=name,offset=offset,node_hash=f'{node:08x}',parent_hash=f'{key:08x}',field_count=struct.unpack_from('<I',data,offset+4)[0],status='structural parent-reference candidate; balance properties not decoded'))
        if refs:
            dest=out/'native_blueprints'/path.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
            blueprints.append(dict(source=str(path),copy=dest.relative_to(out).as_posix(),sha256=hashlib.sha256(data).hexdigest(),references=refs))
    inventory=json.loads((base/'blender_library/manifest.json').read_text())['player_weapon_inventory'];families=[]
    for name,(class_,notes) in FAMILIES.items():
        related=[r for r in records if r['model']==name]
        source=next((r for r in inventory if r['namespace']=='chars__actors_pc' and r['name'].lower()==name.lower()),None)
        tokens={'Artillery_weapon':['artillery'],'Main_Gun':['/cannon/'],'Pistol_Weapon':['/pistol/'],'Autocannon':['auto_cannon','autocannon'],'Gatling_Gun':['machine_gun','gattling','gatling']}.get(name,[name.lower()])
        sounds=[str(p.relative_to(base/'audio_pcm')) for ns in ('chars__actors_pc','sounds__gmsnd_asr') for p in (base/'audio_pcm'/ns).rglob('*.wav') if any(token in p.as_posix().lower() for token in tokens)]
        families.append(dict(name=name,bzr_class_candidate=class_,assessment=notes,native_model=source,controllers=related,audio_candidates=sounds,
            native_damage=None,native_fire_rate=None,native_projectile_speed=None,native_ammo=None,numeric_balance_status='unresolved; no values invented from animation durations or untyped words'))
    schema=None
    if args.executable:
        schema,asm=executable_schema(args.executable.resolve());(out/'weapon_reader.asm').write_text('\n'.join(asm));(out/'weapon_property_schema.json').write_text(json.dumps(schema,indent=2))
    result=dict(families=families,native_controller_records=records,native_blueprints=blueprints,
                executable_schema='weapon_property_schema.json' if schema else None,
                limitations=['Recovered controller references and schema are not complete Gold numeric weapon definitions','Blueprint field layouts/inheritance and weapon balance still need decoding','BZR class candidates are mechanics approximations, not verified Gold gameplay matches'])
    (out/'weapon_definitions.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'recreation_candidates.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['family','bzr_class_candidate','native_controller_records','audio_candidates','numeric_balance_status','assessment'])
        for row in families:writer.writerow([row['name'],row['bzr_class_candidate'],len(row['controllers']),len(row['audio_candidates']),row['numeric_balance_status'],row['assessment']])
    (out/'README.md').write_text('# Gold weapon definition research\n\n[weapon_definitions.json](weapon_definitions.json) retains decoded model/clip controller links, copied raw records, blueprint parent-reference candidates, audio candidates and conservative BZR mechanic mappings. [recreation_candidates.csv](recreation_candidates.csv) is the comparison table.\n\nThe executable property reader establishes native fields such as RateOfFire, bullets per shot, range and burst configuration. Its fallback constants are defaults for missing properties, not recovered weapon-specific tuning. Damage, ROF, speed, ammo and inherited blueprint values remain unresolved. Fire animation duration is not a firing interval.\n\nThe best first recreation is an autocannon using BZR machinegun behavior and Gold audio/model. The optional bzgauto.odf prototype uses stock minigun ballistics and shot delay, explicitly not Gold balance. Charge, heat, reload, guidance, ring animations and effects require additional controllers or supported BZR equivalents.\n',encoding='utf-8')
    print(json.dumps(dict(families=len(families),controller_records=len(records),blueprint_files=len(blueprints),schema_fields=len(schema['properties']) if schema else 0),indent=2))


if __name__=='__main__':main()
