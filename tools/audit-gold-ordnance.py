"""Compare exported Gold projectile axes/origins with real BZR/ISDFC meshes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from gold_weapon_geometry import orientation_landmarks


def mesh_geometry(path):
    from bzrmodelporter.ogremesh_serializer import MeshSerializer
    with path.open('rb') as stream:mesh=MeshSerializer(stream).read()
    positions=[];faces=[]
    for sub in mesh.submeshes():
        vd=sub.vertex_data if sub.vertex_data is not None else mesh.shared_vertex_data
        element=vd.vertex_declaration.get_position_vertex_element();buffer=vd.get_vertex_buffer(element.source)
        values=np.ndarray((vd.vertex_count,3),dtype='<f4',buffer=buffer.buffer,offset=element.offset,strides=(buffer.vertex_size,4)).copy()
        faces.extend((np.array(sub.index_array).reshape(-1,3)+len(positions)).tolist());positions.extend(values)
    return np.array(positions),np.array(faces)


def metrics(points):
    low=points.min(0);high=points.max(0);span=high-low;center=(high+low)/2
    _,vectors=np.linalg.eigh(np.cov(points.T));axis=vectors[:,-1]
    if axis[2]<0:axis=-axis
    return dict(bounds=[low.tolist(),high.tolist()],span=span.tolist(),center=center.tolist(),principal_axis=axis.tolist(),axis_error_degrees=float(np.degrees(np.arccos(np.clip(axis[2],-1,1)))),z_is_longest=bool(np.argmax(span)==2),nose_plane_z=float(high[2]),tail_plane_z=float(low[2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mod-root',type=Path,required=True);parser.add_argument('--game-root',type=Path,required=True);parser.add_argument('--toolkit-root',type=Path,default=Path(__file__).resolve().parents[2]/'BZ98RBlenderToolKit');parser.add_argument('--reference-root',type=Path,default=Path(__file__).resolve().parents[2]/'ISDF_Chronicles-geo-winding');args=parser.parse_args()
    sys.path.insert(0,str(args.toolkit_root/'bz98tools'));root=args.mod_root.resolve();out=root.parent/'review';out.mkdir(exist_ok=True);receipts=json.loads((root.parent/'projectiles.json').read_text());results=[];plots=[]
    stock=args.game_root/'BZ_ASSETS/common/models'
    for name,label in [('gatst00','BZR AT-Stabber'),('ghorn00','BZR Hornet'),('gshad00','BZR Shadower'),('ghartl00','BZR mortar')]:
        path=stock/(name+'.mesh');points,faces=mesh_geometry(path);m=metrics(points)
        if not m['z_is_longest']:raise ValueError('Reference forward axis not longitudinal: '+name)
        row=dict(id=name,label=label,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kind='installed BZR reference',**m);results.append(row);plots.append((row,points,faces))
    path=args.reference_root/'iomshd00.mesh'
    if path.exists():
        points,faces=mesh_geometry(path);row=dict(id='iomshd00',label='ISDFC Hellfire',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kind='ISDFC custom reference',**metrics(points));results.append(row);plots.append((row,points,faces))
    for source in receipts:
        path=root/(source['id']+'.mesh');points,faces=mesh_geometry(path);m=metrics(points);doc=json.loads(Path(source['source']).read_text());landmarks=orientation_landmarks(doc);checks=[]
        for item in landmarks:
            if item['name'].lower().startswith('socket:'):
                # A thrust socket's local Z can be sideways (native Missile).
                # Its location is the independent rear landmark; do not rotate
                # the whole projectile to match an arbitrary socket basis.
                position=(np.array(item['position'])-source['center'])*source['scale']
                if position[2]>=0:raise ValueError('Rear thrust anchor lies ahead of center: '+source['id'])
                checks.append(dict(name=item['name'],rear_position=position.tolist(),rear_z_negative=True,socket_basis_forward=item['forward'],basis_is_flight_axis=False))
            else:
                error=float(np.degrees(np.arccos(np.clip(item['forward'][2],-1,1))))
                if error>.1:raise ValueError('Named front/rear reversed or sideways: '+source['id'])
                checks.append(dict(name=item['name'],forward=item['forward'],error_degrees=error))
        if not checks:raise ValueError('No independent nose/tail landmark: '+source['id'])
        if not m['z_is_longest'] or m['axis_error_degrees']>2 or np.max(np.abs(m['center']))>1e-5:raise ValueError('Uncentered/sideways projectile: '+source['id'])
        row=dict(id=source['id'],label='Gold '+source['source_model'],kind='Gold export',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),landmarks=checks,**m);results.append(row);plots.append((row,points,faces))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig,axes=plt.subplots((len(plots)+3)//4,4,figsize=(17,11),squeeze=False);fig.suptitle('Ordnance orientation: side projection (Z forward, Y up) | actual exported mesh vertices',fontsize=14)
    for ax,(row,points,faces) in zip(axes.flat,plots):
        length=np.ptp(points[:,2]);center=np.array(row['center']);p=(points[:,[2,1]]-center[[2,1]])/length;edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);ax.add_collection(LineCollection(p[edges],colors='#2587bb' if row['kind']=='Gold export' else '#555555',linewidths=.45));ax.autoscale();ax.set_aspect('equal');ax.plot(-center[2]/length,-center[1]/length,'r+',markersize=7);ax.annotate('+Z forward',xy=(.55,-.22),xytext=(-.35,-.22),arrowprops=dict(arrowstyle='->'),fontsize=8);ax.set_xlim(-.72,.72);ax.set_ylim(-.35,.35);ax.set_title(row['id']+' | '+row['label'],fontsize=9);ax.set_xlabel(f"length {row['span'][2]:.3f} m; axis error {row['axis_error_degrees']:.3f} deg",fontsize=8);ax.set_xticks([]);ax.set_yticks([])
    for ax in list(axes.flat)[len(plots):]:ax.axis('off')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'ordnance_orientation.png',dpi=150);plt.close(fig)
    result=dict(passed=True,reference_meshes=sum(r['kind']!='Gold export' for r in results),gold_meshes=len(receipts),forward_axis=[0,0,1],up_axis=[0,1,0],origin_policy='Gold AABB center, like the centered stock projectile references; no claim of recovered native Gold spawn offset',models=results,scope='Geometry, source landmarks and installed reference comparison; no native firing/collision test')
    (root.parent/'orientation_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2))


if __name__=='__main__':main()
