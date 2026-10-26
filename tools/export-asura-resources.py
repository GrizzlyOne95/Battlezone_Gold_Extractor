#!/usr/bin/env python3
"""Recover shader programs, reflection, material records, volume LUTs and dependencies.

Raw sources stay authoritative. Unknown fields are retained without invented semantics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import mmap
from pathlib import Path
import re
import shutil
import struct

import numpy as np
from PIL import Image

from asura_scene import name_hash, padded_name, read_materials
from asura_shaders import WindowsCompiler, read_dxbc, read_shader_bank
from asura_materials import read_material_table


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def texture_info(data):
    if data[:4] != b'DDS ':
        return None
    if len(data) < 128 or struct.unpack_from('<I', data, 4)[0] != 124:
        raise ValueError('Invalid DDS header')
    flags, height, width, pitch, depth, mips = struct.unpack_from('<6I', data, 8)
    fourcc=data[84:88].decode('ascii','backslashreplace')
    caps2=struct.unpack_from('<I',data,112)[0]
    result=dict(width=width,height=height,depth=depth,mip_count=max(mips,1),fourcc=fourcc,caps2=f'{caps2:08x}',is_cubemap=bool(caps2&0x200),is_volume=bool(caps2&0x200000),header_bytes=128)
    if fourcc=='DX10':
        if len(data)<148:raise ValueError('Truncated DDS DX10 header')
        fmt,dimension,misc,array,misc2=struct.unpack_from('<5I',data,128)
        result.update(dxgi_format=fmt,resource_dimension=dimension,array_size=array,misc_flags=misc,misc_flags2=misc2,is_cubemap=bool(misc&4),is_volume=dimension==4,header_bytes=148)
    return result


def read_volumes(data):
    if len(data)<4:raise ValueError('Missing VTEX count')
    count=struct.unpack_from('<I',data)[0]
    if count>65536:raise ValueError('Invalid VTEX count')
    cursor=4;records=[]
    for index in range(count):
        if cursor+16>len(data):raise ValueError('Truncated volume descriptor')
        key,width,height,depth=struct.unpack_from('<4I',data,cursor);cursor+=16
        size=width*height*depth*4
        if not all(0<n<=4096 for n in (width,height,depth)) or cursor+size>len(data):raise ValueError('Invalid volume texture bounds')
        records.append(dict(hash=key,width=width,height=height,depth=depth,offset=cursor,size=size,payload=data[cursor:cursor+size]))
        cursor+=size
    if cursor!=len(data):raise ValueError('Unparsed volume texture tail')
    return records


def scan_dxbc(path):
    if not path.stat().st_size:return []
    found=[]
    with path.open('rb') as stream:
        head=stream.read(12)
        if head[:4] in (b'DDS ',b'RIFF',b'FORM',b'fLaC',b'OggS') or head.startswith(b'\x89PNG'):
            return []
        with mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as data:
            cursor=0
            while True:
                offset=data.find(b'DXBC',cursor)
                if offset<0:break
                try:
                    meta=read_dxbc(data,offset)
                    if not meta['profile']:raise ValueError('No program chunk')
                    found.append((offset,bytes(data[offset:offset+meta['size']]),meta))
                    cursor=offset+meta['size']
                except (ValueError,UnicodeError,struct.error):cursor=offset+4
    return found


def extract(raw_root, output, game_root=None, scan_all=True):
    output.mkdir(parents=True,exist_ok=True)
    compiler=None
    try:compiler=WindowsCompiler()
    except OSError:pass
    program_metadata={};sources=[];banks=[];errors=[]
    def shader(bytecode,meta):
        key=meta['sha256']
        if key in program_metadata:return key
        target=output/'shaders/programs'/f'{key}.dxbc'
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(bytecode)
        record=dict(meta,bytecode=str(target.relative_to(output).as_posix()),reflection_status='unavailable',disassembly_status='unavailable')
        if compiler:
            try:
                reflected=compiler.reflect(bytecode)
                assembly=compiler.disassemble(bytecode)
                asm_path=target.with_suffix('.asm');asm_path.write_text(assembly,encoding='utf-8')
                reflection_path=target.with_suffix('.reflection.json');save_json(reflection_path,reflected)
                record.update(reflection=str(reflection_path.relative_to(output).as_posix()),disassembly=str(asm_path.relative_to(output).as_posix()),reflection_status='ok',disassembly_status='ok',resource_names=[r['Name'] for r in reflected['resources']],constant_buffers=[dict(name=cb['Name'],size=cb['Size']) for cb in reflected['constant_buffers']])
            except Exception as error:
                record['compiler_error']=str(error)
                errors.append(dict(kind='shader_compiler',sha256=key,error=str(error)))
        save_json(target.with_suffix('.json'),record)
        program_metadata[key]=record
        return key
    bank_paths=sorted(raw_root.glob('*/export time*shaders'))
    for p in bank_paths:
        relative=p.relative_to(raw_root).as_posix()
        try:
            entries,tail=read_shader_bank(p.read_bytes(),particle='pfx' in p.name.lower())
            record=dict(source=relative,entries=[],tail_hex=tail.hex(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for entry in entries:
                key=shader(entry.bytecode,entry.metadata)
                item=dict(source=relative,index=entry.index,offset=entry.offset,size=len(entry.bytecode),permutation=f'{entry.permutation:016x}' if entry.permutation is not None else None,sha256=key,profile=entry.metadata['profile'])
                record['entries'].append(item);sources.append(item)
            save_json(output/'shaders/banks'/p.parent.name/(p.name.replace(' ','_')+'.json'),record)
            banks.append(record)
        except Exception as error:errors.append(dict(kind='shader_bank',source=relative,error=str(error)))
    known={(row['source'],row['offset']) for row in sources}
    if scan_all:
        paths=sorted(p for p in raw_root.rglob('*') if p.is_file() and p not in bank_paths)
        if game_root:
            paths.extend(sorted(p for p in game_root.rglob('*') if p.suffix.lower() in ('.exe','.dll')))
        def scan(path):return path,scan_dxbc(path)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for p,hits in pool.map(scan,paths):
                relative=p.relative_to(raw_root).as_posix() if p.is_relative_to(raw_root) else 'installation/'+p.relative_to(game_root).as_posix()
                for offset,bytecode,meta in hits:
                    if (relative,offset) in known:continue
                    key=shader(bytecode,meta)
                    sources.append(dict(source=relative,index=None,offset=offset,size=len(bytecode),permutation=None,sha256=key,profile=meta['profile']))
    save_json(output/'shaders/index.json',dict(programs=program_metadata,sources=sources,banks=len(banks)))
    print(f'Shaders: {len(sources)} occurrences, {len(program_metadata)} unique programs',flush=True)

    # Texture hashes and header data retain mip/cubemap/array distinctions.
    textures=defaultdict(list);texture_rows=[];volume_rows=[]
    all_files=sorted(p for p in raw_root.rglob('*') if p.is_file())
    def inspect_texture(path):
        with path.open('rb') as f:head=f.read(148)
        return path,texture_info(head)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for p,info in pool.map(inspect_texture,all_files):
            if info:
                relative=p.relative_to(raw_root);ns=relative.parts[0]
                source=relative.as_posix();key=name_hash('/'+Path(*relative.parts[1:]).with_suffix('').as_posix().removeprefix('graphics/'))
                textures[key].append(source)
                texture_rows.append(dict(source=source,namespace=ns,native_hash=f'{key:08x}',bytes=p.stat().st_size,**info))
    for p in sorted(raw_root.glob('*/VTEX_chunk/*.dat')):
        try:
            for item in read_volumes(p.read_bytes()):
                payload=item.pop('payload');key=hashlib.sha256(payload).hexdigest();item['sha256']=key
                folder=output/'volume_textures'/p.parent.parent.name/f'{item["hash"]:08x}'
                folder.mkdir(parents=True,exist_ok=True)
                raw_path=folder/'channels_u8.bin';raw_path.write_bytes(payload)
                array=np.frombuffer(payload,dtype=np.uint8).reshape(item['depth'],item['height'],item['width'],4)
                np.save(folder/'channels_u8.npy',array)
                # Explicit channel interpretation for inspection only; raw bytes are authoritative.
                grid=Image.new('RGBA',(item['width']*math.ceil(math.sqrt(item['depth'])),item['height']*math.ceil(item['depth']/math.ceil(math.sqrt(item['depth'])))))
                columns=math.ceil(math.sqrt(item['depth']))
                for z in range(item['depth']):grid.paste(Image.fromarray(array[z]),((z%columns)*item['width'],(z//columns)*item['height']))
                grid.save(folder/'channel0123_slice_preview.png')
                item.update(source=p.relative_to(raw_root).as_posix(),native_hash=f'{item["hash"]:08x}',raw_file=raw_path.relative_to(output).as_posix(),channel_semantics='unverified four-byte channels; preview interprets channels 0,1,2,3 as RGBA',shape=[item['depth'],item['height'],item['width'],4])
                save_json(folder/'metadata.json',item);volume_rows.append(item)
        except Exception as error:errors.append(dict(kind='volume_texture',source=str(p),error=str(error)))
    save_json(output/'textures/index.json',dict(dds=texture_rows,volume_textures=volume_rows))

    materials=[];material_lookup=defaultdict(list);field_layouts={};response_luts=[];unique_response_luts=set()
    for key,program in program_metadata.items():
        if program.get('reflection_status')!='ok':continue
        reflection=json.loads((output/program['reflection']).read_text())
        for buffer in reflection['constant_buffers']:
            for variable in buffer['variables']:
                typename=variable['type'].get('Name')
                if typename and ('Material' in typename or 'SkinPS' in buffer['Name']):
                    signature=hashlib.sha256(json.dumps(variable['type'],sort_keys=True).encode()).hexdigest()
                    field_layouts[signature]=dict(name=typename,buffer=buffer['Name'],variable=variable['Name'],type=variable['type'],example_program=key)
    save_json(output/'materials/reflected_parameter_layouts.json',field_layouts)
    for p in sorted(raw_root.glob('*/MARE_chunk/*.dat')):
        try:
            table=read_material_table(p.read_bytes());ns=p.parent.parent.name;rows=[]
            for key,lut in table['response_luts'].items():
                raw=lut['payload'];digest=lut['sha256'];folder=output/'materials/response_luts';folder.mkdir(parents=True,exist_ok=True)
                binary=folder/(digest+'.float32.bin');binary.write_bytes(raw)
                samples=np.frombuffer(raw,dtype='<f4').reshape(256,4)
                if not np.isfinite(samples).all():raise ValueError('Nonfinite material response LUT')
                if digest not in unique_response_luts:
                    save_json(folder/(digest+'.json'),dict(sha256=digest,shape=[256,4],dtype='little-endian float32',samples=samples.tolist(),semantics='Native response-table channels; view-dependent shader sampling differs by permutation. Tail samples retain metadata; not an albedo palette.'))
                unique_response_luts.add(digest)
                response_luts.append(dict(namespace=ns,source=p.relative_to(raw_root).as_posix(),hash=f'{key:08x}',native_index=lut['index'],source_offset=lut['offset'],sha256=digest,binary=binary.relative_to(output).as_posix()))
            for key,record in table['materials'].items():
                words=record['raw_words'];raw=struct.pack('<89I',*words)
                record=dict(record,source=p.relative_to(raw_root).as_posix(),namespace=ns,raw_hex=raw.hex(),raw_float32=[value if math.isfinite(value) else None for value in struct.unpack('<89f',raw)],exact_texture_hash_matches=[])
                for byte in range(8,356,4):
                    value=words[byte//4]
                    if value in textures:record['exact_texture_hash_matches'].append(dict(byte_offset=byte,hash=f'{value:08x}',sources=textures[value],semantic='unverified; exact hash match'))
                material_lookup[key].append(dict(source=record['source'],namespace=ns,record_offset=record['offset']))
                materials.append(record);rows.append(record)
            save_json(output/'materials/tables'/ns/(p.stem+'.json'),rows)
        except Exception as error:errors.append(dict(kind='material_table',source=str(p),error=str(error)))
    save_json(output/'materials/index.json',dict(records=materials,lookup={f'{key:08x}':value for key,value in material_lookup.items()}))
    save_json(output/'materials/response_luts/index.json',dict(records=response_luts,unique_payloads=len(unique_response_luts)))

    # Every non-media chunk gets a byte-level inventory. VFX/UI names and paths are
    # indexed as strings, not mislabeled as a complete behavior decoder.
    family_rows=defaultdict(list);effect_groups=defaultdict(list);texture_animations=[]
    def inspect_chunk(path):
        data=path.read_bytes();relative=path.relative_to(raw_root).as_posix()
        strings=[]
        for match in re.finditer(rb'[A-Za-z_/\\][A-Za-z0-9_./\\:() -]{3,}',data):
            text=match.group().decode('ascii')
            if len(text)>=5:strings.append(dict(offset=match.start(),value=text))
        row=dict(source=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),first_u32=struct.unpack_from('<I',data)[0] if len(data)>=4 else None,strings=strings)
        return path,row,data
    chunk_paths=[p for p in all_files if p.parent.name.endswith('_chunk')]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for p,row,data in pool.map(inspect_chunk,chunk_paths):
            family=p.parent.name.removesuffix('_chunk');family_rows[family].append(row)
            if family in ('FXST','FXPT','FXET','FXTT'):
                try:
                    owner,cursor=padded_name(data,4)
                    component=None
                    if family in ('FXST','FXPT'):component,_=padded_name(data,cursor)
                    row.update(effect=owner,component=component,native_id=f'{row["first_u32"]:08x}')
                    effect_groups[(p.parent.parent.name,owner)].append(row)
                except (ValueError,UnicodeError):pass
            if family=='TXAN':
                try:
                    name,cursor=padded_name(data,0);category,cursor=padded_name(data,cursor)
                    row.update(name=name,category=category,payload_offset=cursor,payload_hex=data[cursor:].hex(),decode_status='header and exact payload retained; curve semantics unresolved')
                    texture_animations.append(row)
                except (ValueError,UnicodeError):pass
    for family,rows in family_rows.items():save_json(output/'native_chunks'/f'{family}.json',rows)
    save_json(output/'effects/index.json',dict(groups=[dict(namespace=namespace,name=name,components=rows) for (namespace,name),rows in effect_groups.items()],texture_animations=texture_animations,decode_status='named dependencies recovered; runtime simulation and curve semantics unresolved'))
    summary=dict(shader_banks=len(banks),shader_occurrences=len(sources),unique_shader_programs=len(program_metadata),profiles=dict(Counter(record['profile'] for record in program_metadata.values())),disassembled=sum(p['disassembly_status']=='ok' for p in program_metadata.values()),reflected=sum(p['reflection_status']=='ok' for p in program_metadata.values()),dds_textures=len(texture_rows),cubemap_textures=sum(p['is_cubemap'] for p in texture_rows),dds_volume_textures=sum(p['is_volume'] for p in texture_rows),vtex_volumes=len(volume_rows),unique_vtex_payloads=len({r['sha256'] for r in volume_rows}),material_records=len(materials),reflected_material_layouts=len(field_layouts),native_chunk_families=len(family_rows),native_chunk_files=len(chunk_paths),effect_groups=len(effect_groups),texture_animation_chunks=len(texture_animations),errors=errors,limitations=['Original HLSL source is not reconstructed; DXBC and disassembly are preserved.','Material parameter layouts come from shader reflection; assigning serialized MARE fields to all reflected constants remains unresolved.','VFX/UI/texture-animation behavior is indexed and byte-preserved, not fully simulated.','Volume channel meanings are unverified; raw 4-channel volumes remain authoritative.'])
    summary.update(material_response_luts=len(response_luts),unique_material_response_luts=len(unique_response_luts))
    summary['limitations'][1]='Native default tint, material model, emission and selected constants are decoded. Entity overrides, full LUT semantics and exact BRDF reproduction remain unresolved.'
    save_json(output/'resource-summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return int(bool(errors))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--game-root',type=Path)
    parser.add_argument('--banks-only',action='store_true',help='Skip the additional embedded shader scan')
    args=parser.parse_args()
    return extract(args.input_root.resolve(),args.output_root.resolve(),args.game_root.resolve() if args.game_root else None,not args.banks_only)


if __name__=='__main__':raise SystemExit(main())
