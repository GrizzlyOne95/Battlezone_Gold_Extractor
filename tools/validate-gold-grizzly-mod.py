"""Validate the standalone mod package with shared readers, without launching BZR."""
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
from bzr_materials import validate_stock_inheritance

import numpy as np


def transform_matrix(transform):
    return np.array([[transform.rx,transform.ux,transform.fx,transform.px],
                     [transform.ry,transform.uy,transform.fy,transform.py],
                     [transform.rz,transform.uz,transform.fz,transform.pz],[0,0,0,1]],dtype=float)


def skeleton_worlds(skeleton):
    result={}
    def world(bone):
        if bone.handle not in result:
            matrix=transform_matrix(bone.orientation.to_transform());matrix[:3,3]=[bone.position.x,bone.position.y,bone.position.z]
            matrix[:3,:3]*=np.array([bone.scale.x,bone.scale.y,bone.scale.z])[None,:]
            result[bone.handle]=world(bone.parent)@matrix if bone.parent else matrix
        return result[bone.handle]
    for bone in skeleton.bones():world(bone)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mod-root',type=Path,required=True);parser.add_argument('--game-root',type=Path,required=True)
    parser.add_argument('--toolkit-root',type=Path,default=Path(__file__).resolve().parents[2]/'BZ98RBlenderToolKit');parser.add_argument('--toolbox-root',type=Path,default=Path(__file__).resolve().parents[2]/'Battlezone98_Modding_Toolbox')
    args=parser.parse_args();root=args.mod_root.resolve();game=args.game_root.resolve()
    sys.path.insert(0,str(args.toolkit_root/'bz98tools'));sys.path.insert(0,str(args.toolbox_root))
    from bzrmodelporter.ogremesh_serializer import MeshSerializer
    from bzrmodelporter.ogreskeleton_serializer import SkeletonSerializer
    from bzrmodelporter.bzbwd2_serializer import VDFSerializer
    from battlezone.archives.zfs import ZFSArchive
    from battlezone.meshes.legacy import read_vdf
    receipt=json.loads((root.parent/'build.json').read_text());hashes=json.loads((root.parent/'runtime_sha256.json').read_text())
    if not receipt.get('uv_convention','').startswith('Blender U,V -> Ogre U,1-V'):raise ValueError('Port does not declare the required Ogre UV conversion')
    ini=configparser.ConfigParser();ini.read(root/'goldgrizzly.ini')
    if ini['WORKSHOP']['mapType'].strip('"')!='mod':raise ValueError('Not a standalone mod INI')
    for name,digest in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Runtime file hash mismatch: '+name)
    archive=ZFSArchive(game/'bzone.zfs');original=archive.read('avtank.vdf');stock=read_vdf(original);target=read_vdf((root/'avtank.vdf').read_bytes())
    originals={(p.band,p.slot):p for p in stock.parts};changed={'AGR11POV',*(m['part'] for m in receipt['muzzle_mappings'])}
    if len(stock.parts)!=len(target.parts) or stock.sequences!=target.sequences or stock.tracks!=target.tracks:raise ValueError('Stock VDF structure or animation changed')
    preserved=0
    for part in target.parts:
        before=originals[part.band,part.slot]
        if part.band==0 and part.name in changed:
            if (before.name,before.klass,before.flags)!=(part.name,part.klass,part.flags):raise ValueError('Changed gameplay part type')
        else:
            if part!=before:raise ValueError('Unintended VDF part edit: '+part.name)
            preserved+=1
    before=VDFSerializer(io.BytesIO(original)).read()
    with (root/'avtank.vdf').open('rb') as stream:after=VDFSerializer(stream).read()
    # The existing serializer supplies the collision representation; no codec here.
    def plain(value):
        if isinstance(value,(str,int,float,bool,type(None))):return value
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
        if hasattr(value,'__dict__'):return {k:plain(v) for k,v in value.__dict__.items()}
        raise TypeError(type(value))
    if plain(before.collision_planes)!=plain(after.collision_planes):raise ValueError('Collision planes changed')
    material_text=(root/'goldgrizzly.material').read_text();definitions=set(re.findall(r'^material\s+(\S+)',material_text,re.M))
    material_imports=validate_stock_inheritance(material_text,(game/'BZ_ASSETS_CORE/common/BZ_MATERIALS/BZBase.material').read_text())
    textures=set(re.findall(r'set_texture_alias\s+\S+\s+(\S+)',material_text))
    for name in textures:
        if name!='flat_n.png' and not (root/name).exists():raise ValueError('Missing texture '+name)
    results=[];muzzles=[]
    for base in ('avtank','avtank_cockpit'):
        with (root/(base+'.mesh')).open('rb') as stream:mesh=MeshSerializer(stream).read()
        with (root/(base+'.skeleton')).open('rb') as stream:skeleton=SkeletonSerializer(stream).read()
        if mesh.skeleton_name!=base+'.skeleton':raise ValueError('Wrong skeleton link')
        worlds=skeleton_worlds(skeleton);vertices=triangles=0
        for sub in mesh.submeshes():
            if sub.material_name not in definitions:raise ValueError('Missing material '+sub.material_name)
            if sub.vertex_data.get_vertex_buffer(0).vertex_size!=24 or sub.vertex_data.get_vertex_buffer(1).vertex_size!=12:raise ValueError('Wrong BZR buffer layout')
            data=np.frombuffer(sub.vertex_data.get_vertex_buffer(0).buffer,dtype='<f4').reshape(-1,6)
            if not np.isfinite(data).all() or np.max(np.abs(np.linalg.norm(data[:,3:6],axis=1)-1))>1e-4:raise ValueError('Invalid geometry or normals')
            if int(np.max(sub.index_array))>=len(data):raise ValueError('Out of range indices')
            weights=list(sub.bone_assignments())
            if len(weights)!=len(data) or any(w.weight!=1 or w.bone_index not in worlds for w in weights):raise ValueError('Invalid rigid control bindings')
            vertices+=len(data);triangles+=sub.index_count//3
        pov=worlds[skeleton.get_bone_by_name('AGR11POV').handle][:3,3]
        np.testing.assert_allclose(pov,receipt['authored_eye'],atol=2e-6)
        for mapping in receipt['muzzle_mappings']:
            bone=skeleton.get_bone_by_name(mapping['part']);actual=worlds[bone.handle][:3,3]
            np.testing.assert_allclose(actual,mapping['world'],atol=2e-6)
            part=next(p for p in target.parts if p.band==0 and p.name==mapping['part'])
            assert part.parent=='AGR11TUR' and bone.parent.name=='AGR11TUR'
            np.testing.assert_allclose([part.matrix[9],part.matrix[10],part.matrix[11]],[bone.position.x,bone.position.y,bone.position.z],atol=2e-6)
            angle=math.degrees(math.acos(np.clip(np.dot(mapping['source_forward'],[0,0,1]),-1,1)))
            if angle>.01:raise ValueError('Source muzzle axis differs from runtime forward')
            muzzles.append(dict(model=base,part=part.name,error=float(np.max(np.abs(actual-mapping['world']))),source_axis_error_degrees=angle))
        results.append(dict(model=base,vertices=vertices,triangles=triangles,submeshes=len(list(mesh.submeshes())),bones=len(list(skeleton.bones()))))
    weapon=configparser.ConfigParser();weapon.read_string('\n'.join(line for line in (root/'bzgauto.odf').read_text().splitlines() if not line.startswith('//')))
    assert weapon['WeaponClass']['classLabel'].strip('"')=='machinegun' and weapon['WeaponClass']['ordName'].strip('"')=='bullet1'
    assert float(weapon['CannonClass']['shotDelay'])==.2
    with wave.open(str(root/'bzgauto.wav')) as audio:
        assert (audio.getnchannels(),audio.getsampwidth(),audio.getframerate(),audio.getcomptype())==(1,2,22050,'NONE')
    if (root/'avtank.odf').exists():raise ValueError('Default Grizzly loadout must remain untouched')
    with zipfile.ZipFile(root.parent/'goldgrizzly.zip') as bundle:
        if bundle.testzip():raise ValueError('Corrupt ZIP')
        for name,digest in hashes.items():
            if hashlib.sha256(bundle.read('goldgrizzly/'+name)).hexdigest()!=digest:raise ValueError('ZIP/runtime mismatch')
    result=dict(passed=True,runtime_manifest_sha256=hashlib.sha256((root.parent/'runtime_sha256.json').read_bytes()).hexdigest(),models=results,muzzles=muzzles,stock_vdf_parts_preserved=preserved,collision_unchanged=True,
                runtime_files=len(hashes),texture_dependencies=len(textures),material_imports=material_imports,missing_dependencies=[],zip_verified=True,
                optional_weapon_equipped=False,scope='offline binary/package consistency; native rendering, collision fit and gameplay unqualified')
    (root.parent/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
