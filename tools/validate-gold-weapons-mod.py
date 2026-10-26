"""Offline dependency, ODF graph, shared binary and package validation."""
import argparse
import configparser
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import wave
import zipfile

import numpy as np
from PIL import Image
from gold_weapons import parse_odf, verify_render_graph, recipes, read_gold_text
from bzr_materials import validate_stock_inheritance


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mod-root',type=Path,required=True);parser.add_argument('--game-root',type=Path,required=True);parser.add_argument('--toolkit-root',type=Path,default=Path(__file__).resolve().parents[2]/'BZ98RBlenderToolKit');args=parser.parse_args()
    root=args.mod_root.resolve();game=args.game_root.resolve();receipt=json.loads((root.parent/'build.json').read_text());hashes=json.loads((root.parent/'runtime_sha256.json').read_text());projectiles=json.loads((root.parent/'projectiles.json').read_text())
    if set(hashes)!={p.name for p in root.iterdir() if p.is_file()}:raise ValueError('Runtime manifest does not cover package exactly')
    for name,expected in hashes.items():
        if Path(name).name!=name or digest(root/name)!=expected:raise ValueError('Runtime hash mismatch: '+name)
    ini=configparser.ConfigParser();ini.read(root/'goldweapons.ini')
    if ini['WORKSHOP']['mapType'].strip('"')!='mod':raise ValueError('Not a standalone mod')
    docs={p.stem.lower():parse_odf(p.read_text()) for p in root.glob('*.odf')};roots=[];checked_audio=[];textures=set()
    rows=receipt['weapons']
    if {r['family'] for r in rows}!={r['family'] for r in recipes()} or len(rows)!=len(recipes()):raise ValueError('Weapon family missing/duplicated')
    cue_count=0
    for row in rows:
        w=docs[row['weapon']]['weaponclass'];o=docs[row['ordnance']]['ordnanceclass'];x=docs[row['explosion']]['explosionclass'];p=docs[row['pickup']]['weaponpowerupclass']
        expected_class=parse_odf((game/'StockODFFiles'/(row['weapon_template']+'.odf')).read_text())['weaponclass']['classlabel']
        if w['classlabel']!=expected_class or w['ordname']!=row['ordnance'] or p['weaponname']!=row['weapon']:raise ValueError('Weapon/pickup link mismatch')
        for key in ('xplground','xplvehicle','xplbuilding','xplexpire'):
            if o[key]!=row['explosion']:raise ValueError('Missing impact/expiry connection')
        if float(o['damageconcussion'])!=0 or float(x['damageconcussion'])!=row['blast_damage']:raise ValueError('Blast damage duplicated or missing')
        if o['shotgeometry']!='NULL':
            if o['shotgeometry']!=row['projectile']+'.geo' or not (root/o['shotgeometry']).is_file():raise ValueError('Projectile mesh link missing')
        for key in ('shotspeed','lifespan','ammocost','damageballistic'):
            if not math.isfinite(float(o[key])) or float(o[key])<0:raise ValueError('Invalid ordnance scalar')
        roots.append(o['rendername'])
        count=int(x['particletypes'])
        if {k for k in x if re.fullmatch(r'particleclass\d+',k)}!={'particleclass'+str(i) for i in range(1,count+1)}:raise ValueError('Explosion particle count mismatch')
        roots.extend(x['particleclass'+str(i)] for i in range(1,count+1))
        audio_entries=[(row['fire_audio'],'fire'),(row['impact_audio'],'impact')]+[(c,'cue:'+c['cue']) for c in row.get('audio_cues',[])]
        cue_count+=len(row.get('audio_cues',[]))
        for key,value in audio_entries:
            if key is None:continue
            path=root/key['file']
            if digest(Path(key['source']))!=key['source_sha256']:raise ValueError('Source audio changed')
            with wave.open(str(path),'rb') as wav:
                if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getcomptype())!=(1,2,22050,'NONE') or not wav.getnframes():raise ValueError('Incompatible audio')
                samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(float)/32768
                if not np.any(samples):raise ValueError('Silent converted audio: '+path.name)
                checked_audio.append(dict(weapon=row['weapon'],type=value,file=path.name,seconds=wav.getnframes()/wav.getframerate(),peak=float(np.max(np.abs(samples))),rms=float(np.sqrt(np.mean(samples*samples)))))
        active_fire={row['fire_audio']['file']}|{c['file'] for c in row.get('audio_cues',[]) if c['runtime_connection']!='unassigned; retained native cue for later event/controller work'}
        if w['firesound'] not in active_fire|({'NULL'} if row['kind']=='burst' else set()) or x['explsound']!=(row['impact_audio']['file'] if row['impact_audio'] else 'NULL'):raise ValueError('Audio not connected')
        if row['kind'] in ('charge_beam','burst'):
            charge=docs[row['weapon']]['chargegunclass'];count=int(charge['ordnancecount'])
            for level in range(2,count+1):
                if charge['ordname'+str(level)] not in docs or not 1<=int(charge['salvocount'+str(level)])<=8:raise ValueError('Invalid charge/salvo profile')
                if charge['firesound'+str(level)]!=row['fire_audio']['file']:raise ValueError('Charge release sound disconnected')
        if row['kind']=='flare':
            obj=docs[row['weapon']]['dispenserclass']['objectclass']
            if docs[obj]['gameobjectclass']['classlabel']!='flare' or docs[obj]['flaremineclass']['payloadname']!=row['ordnance']:raise ValueError('Flare decoy graph disconnected')
        if row['kind']=='time_fuse' and (o['classlabel']!='bullet' or float(o['lifespan'])!=.65):raise ValueError('MAG expiry fuse is not on a physical bullet')
    # Also visit render roots of charge-level and cluster child ordnance.
    roots.extend(v['rendername'] for doc in docs.values() for name,v in doc.items() if name=='ordnanceclass' and 'rendername' in v)
    for doc in docs.values():
        if 'quakeblastclass' in doc:
            quake=doc['quakeblastclass'];child=docs[quake['quakeclass']]['ordnanceclass']
            if int(quake['quakecount'])>8 or any(child.get(k,'NULL')!='NULL' for k in ('xplground','xplvehicle','xplbuilding','xplexpire')):raise ValueError('Unbounded cluster child explosion')
    reached=verify_render_graph(docs,roots)
    aliases=receipt.get('tier_aliases',[])
    for alias in aliases:
        original=docs[alias['profile']];actual=docs[alias['weapon']]
        for section,values in original.items():
            for key,value in values.items():
                if section=='weaponclass' and key=='wpnname':continue
                if actual[section][key]!=value:raise ValueError('Tier alias silently changes BZR tuning')
        if docs[alias['pickup']]['weaponpowerupclass']['weaponname']!=alias['weapon']:raise ValueError('Tier pickup mismatch')
    coverage=json.loads((root.parent/'coverage.json').read_text());native_text=read_gold_text(Path(coverage['source']).read_bytes())
    if digest(Path(coverage['source']))!=coverage['source_sha256']:raise ValueError('Native weapon text source changed')
    if {k for k in native_text if k.startswith('WEAPON_DESC_')}-{r['text_key'] for r in rows}:raise ValueError('Native weapon description not covered')
    for ref in reached:
        file,section=ref.split('.',1);values=docs[file][section]
        if 'texturename' in values:
            path=root/values['texturename'];textures.add(path.name)
            if not path.is_file() or len(path.name)>12:raise ValueError('Missing/overlong legacy texture: '+str(path))
            with Image.open(path) as image:image.verify()
            # Gold RGBA cards have bright RGB even where alpha is zero.
            # Reject unmasked additive blending which would expose squares.
            if not values.get('textureblend','').startswith('srcalpha '):raise ValueError('Effect texture alpha is not used: '+ref)
    # Only loadout names change; retain stock physical/gameplay vehicle settings.
    original=parse_odf((game/'StockODFFiles/avtank.odf').read_text());target=docs['avtank']
    for section,values in original.items():
        for key,value in values.items():
            if section=='gameobjectclass' and re.fullmatch(r'weapon(hard|name)[1-4]',key):continue
            if target[section][key]!=value:raise ValueError('Unexpected Grizzly override: '+key)
    for index,(weapon,hard,category) in enumerate([('bgmain','GC1','CANN'),('bgauto','GC2','CANN'),('bgrail','GS1','SPEC'),('bgarty','GM1','MORT')],1):
        g=target['gameobjectclass']
        if g['weaponname'+str(index)]!=weapon or g['weaponhard'+str(index)]!=hard or docs[weapon]['weaponclass']['wpncategory']!=category:raise ValueError('Loadout/category/socket mismatch')
    sys.path.insert(0,str(args.toolkit_root/'bz98tools'))
    from bzrmodelporter.ogremesh_serializer import MeshSerializer
    from bzrmodelporter.ogreskeleton_serializer import SkeletonSerializer
    from bzrmodelporter.bzgeo_serializer import GeoSerializer
    stock_materials=(game/'BZ_ASSETS_CORE/common/BZ_MATERIALS/BZBase.material').read_text()
    material_imports={p.name:validate_stock_inheritance(p.read_text(),stock_materials) for p in root.glob('*.material')}
    material_text='\n'.join(p.read_text() for p in root.glob('*.material'));materials=re.findall(r'\bmaterial\s+([^\s:{]+)',material_text)
    if len(materials)!=len(set(materials)):raise ValueError('Duplicate material declaration')
    for texture in re.findall(r'set_texture_alias\s+\w+\s+(\S+)',material_text):
        if texture=='flat_n.png':
            if not list((game/'BZ_ASSETS_CORE').rglob(texture)):raise ValueError('Stock normal fallback missing')
        elif not (root/texture).is_file():raise ValueError('Material texture missing: '+texture)
    model_results=[]
    for row in projectiles:
        stem=row['id']
        with (root/(stem+'.mesh')).open('rb') as stream:mesh=MeshSerializer(stream).read()
        with (root/(stem+'.skeleton')).open('rb') as stream:skeleton=SkeletonSerializer(stream).read()
        with (root/(stem+'.geo')).open('rb') as stream:geo=GeoSerializer(stream).read()
        if mesh.skeleton_name!=stem+'.skeleton' or {b.name for b in skeleton.bones()}!={stem} or geo.name!=stem:raise ValueError('Model naming/link mismatch')
        points=[];triangles=0
        for sub in mesh.submeshes():
            if sub.material_name not in materials:raise ValueError('Model material missing')
            array=np.frombuffer(sub.vertex_data.get_vertex_buffer(0).buffer,dtype='<f4').reshape(-1,6);points.extend(array[:,:3]);indices=np.array(sub.index_array)
            if not np.isfinite(array).all() or indices.min()<0 or indices.max()>=len(array):raise ValueError('Invalid projectile geometry')
            np.testing.assert_allclose(np.linalg.norm(array[:,3:6],axis=1),1,atol=2e-5)
            uv=np.ndarray((len(array),2),dtype='<f4',buffer=sub.vertex_data.get_vertex_buffer(1).buffer,offset=4,strides=(12,4))
            if not np.isfinite(uv).all():raise ValueError('Nonfinite UV')
            if any(a.bone_index!=0 or a.weight!=1 for a in sub.bone_assignments()):raise ValueError('Projectile control binding lost')
            triangles+=len(indices)//3
        points=np.array(points)
        if abs(np.ptp(points[:,2])-row['target_length'])>1e-4:raise ValueError(f'Projectile scale mismatch: {stem}, got {np.ptp(points[:,2])}, expected {row["target_length"]}')
        if len(geo.face_list)!=triangles:raise ValueError('GEO/mesh topology mismatch')
        geo_points=np.asarray(geo.vertex_pos_buffer).reshape(-1,3)
        def ordered(values):return values[np.lexsort((values[:,2],values[:,1],values[:,0]))]
        np.testing.assert_array_equal(ordered(geo_points),ordered(points))
        for face in geo.faces():
            if len(face.texture_name)>12 or not (root/face.texture_name).exists():raise ValueError('GEO texture dependency missing')
        model_results.append(dict(id=stem,vertices=len(points),triangles=triangles,length=float(np.ptp(points[:,2]))))
    with zipfile.ZipFile(root.parent/'goldweapons.zip') as archive:
        if set(archive.namelist())!={'goldweapons/'+name for name in hashes}:raise ValueError('ZIP file set mismatch')
        for name,expected in hashes.items():
            if hashlib.sha256(archive.read('goldweapons/'+name)).hexdigest()!=expected:raise ValueError('ZIP payload mismatch')
    result=dict(passed=True,runtime_manifest_sha256=digest(root.parent/'runtime_sha256.json'),weapons=len(rows),pickups=len(rows),explosions=len(rows),render_sections=len(reached),effect_textures=len(textures),audio=checked_audio,projectile_models=model_results,runtime_files=len(hashes),zip_verified=True,grizzly_loadout=['bgmain','bgauto','bgrail','bgarty'],scope='Offline only: native game loading, renderer, effects, listening and gameplay still unqualified')
    orientation=json.loads((root.parent/'orientation_validation.json').read_text())
    if not orientation['passed']:raise ValueError('Ordnance reference orientation audit missing')
    for row in orientation['models']:
        if digest(Path(row['path']))!=row['sha256']:raise ValueError('Mesh changed since orientation audit')
    result.update(material_imports=material_imports,tier_aliases=len(aliases),retained_audio_cues=cue_count,native_weapon_descriptions=len(coverage['weapon_descriptions']),reference_orientation_audit=True)
    (root.parent/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
