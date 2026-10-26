"""glTF hierarchy/landmark math used to qualify projectile orientation."""
import numpy as np

GLTF_TO_BZR=np.diag([-1.,1.,-1.]) # shared source Blender import + recorded target axis


def gltf_worlds(document):
    nodes=document['nodes'];parents={c:i for i,n in enumerate(nodes) for c in n.get('children',[])};worlds={}
    def world(i):
        if i in worlds:return worlds[i]
        n=nodes[i]
        if 'matrix' in n:local=np.array(n['matrix'],dtype=float).reshape(4,4).T
        else:
            x,y,z,w=n.get('rotation',[0,0,0,1]);local=np.eye(4)
            local[:3,:3]=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])@np.diag(n.get('scale',[1,1,1]))
            local[:3,3]=n.get('translation',[0,0,0])
        worlds[i]=world(parents[i])@local if i in parents else local;return worlds[i]
    return [world(i) for i in range(len(nodes))]


def orientation_landmarks(document):
    worlds=gltf_worlds(document);nodes=document['nodes'];markers=[]
    for i,node in enumerate(nodes):
        if node['name'].lower()=='socket:thrust':
            m=worlds[i];axis=GLTF_TO_BZR@m[:3,2];axis/=np.linalg.norm(axis)
            markers.append(dict(name=node['name'],position=(GLTF_TO_BZR@m[:3,3]).tolist(),forward=axis.tolist()))
    front=[GLTF_TO_BZR@worlds[i][:3,3] for i,n in enumerate(nodes) if 'frontwing' in n['name'].lower().replace('_','')]
    rear=[GLTF_TO_BZR@worlds[i][:3,3] for i,n in enumerate(nodes) if 'rearwing' in n['name'].lower().replace('_','') and (n['name'].lower().endswith('a') or 'rearwing' in n['name'].lower())]
    if front and rear:
        delta=np.mean(front,axis=0)-np.mean(rear,axis=0);delta/=np.linalg.norm(delta)
        markers.append(dict(name='front_wings minus rear_wings',forward=delta.tolist()))
    body=[i for i,n in enumerate(nodes) if n['name'].lower()=='unguided_rocket_body'];tail=[i for i,n in enumerate(nodes) if n['name'].lower()=='unguided_rocket_tail']
    if body and tail:
        delta=GLTF_TO_BZR@(worlds[body[0]][:3,3]-worlds[tail[0]][:3,3]);delta/=np.linalg.norm(delta)
        markers.append(dict(name='body origin minus named tail',forward=delta.tolist()))
    return markers
