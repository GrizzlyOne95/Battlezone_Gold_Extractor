"""Lossless DDS subresource splitting and inspection environment projection."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import struct
import numpy as np
from PIL import Image


def split_legacy_cubemap(data):
    if len(data)<128 or data[:4]!=b'DDS ':raise ValueError('Missing DDS header')
    height,width=struct.unpack_from('<II',data,12)
    mips=max(1,struct.unpack_from('<I',data,28)[0])
    fourcc=data[84:88];caps2=struct.unpack_from('<I',data,112)[0]
    if not caps2&0x200 or caps2&0xfc00!=0xfc00:raise ValueError('Expected six-face legacy cubemap')
    block={b'DXT1':8,b'DXT3':16,b'DXT5':16,b'ATI1':8,b'ATI2':16}.get(fourcc)
    bits=struct.unpack_from('<I',data,88)[0]
    if not block and (fourcc!=bytes(4) or bits not in (8,16,24,32)):raise ValueError('Unsupported cubemap pixel format')
    sizes=[]
    for mip in range(mips):
        w,h=max(width>>mip,1),max(height>>mip,1)
        sizes.append(max(1,(w+3)//4)*max(1,(h+3)//4)*block if block else w*h*(bits//8))
    chain=sum(sizes)
    if len(data)!=128+6*chain:raise ValueError('Cubemap payload size does not match face/mip layout')
    faces=[]
    for index,name in enumerate(('px','nx','py','ny','pz','nz')):
        header=bytearray(data[:128])
        struct.pack_into('<I',header,112,0)
        struct.pack_into('<I',header,24,0)
        # Retain each complete mip chain in a standard 2D DDS.
        faces.append(dict(name=name,dds=bytes(header)+data[128+index*chain:128+(index+1)*chain],native_offset=128+index*chain,mip_bytes=sizes))
    return faces


def cubemap_latlong(faces,width=1024):
    """Inspection projection using conventional DirectX face order; no HDR invention."""
    arrays=[]
    for face in faces:
        with Image.open(BytesIO(face['dds'])) as image:arrays.append(np.array(image.convert('RGBA')))
    size=arrays[0].shape[0]
    if any(a.shape!=(size,size,4) for a in arrays):raise ValueError('Cube faces must be square and equal')
    lon=(np.arange(width)+.5)/width*2*np.pi-np.pi
    lat=np.pi/2-(np.arange(width//2)+.5)/(width//2)*np.pi
    direction=np.stack(np.broadcast_arrays(np.cos(lat)[:,None]*np.sin(lon),np.sin(lat)[:,None]+np.zeros((1,width)),np.cos(lat)[:,None]*np.cos(lon)),axis=-1)
    x,y,z=direction[...,0],direction[...,1],direction[...,2]
    axis=np.argmax(np.abs(direction),axis=-1);output=np.zeros((width//2,width,4),dtype=np.uint8)
    conditions=[(axis==0)&(x>=0),(axis==0)&(x<0),(axis==1)&(y>=0),(axis==1)&(y<0),(axis==2)&(z>=0),(axis==2)&(z<0)]
    pairs=[(-z,-y),(z,-y),(x,z),(x,-z),(x,-y),(-x,-y)]
    major=np.max(np.abs(direction),axis=-1)
    for i,mask in enumerate(conditions):
        u,v=pairs[i];u=(u[mask]/major[mask]+1)*.5;v=(v[mask]/major[mask]+1)*.5
        ix=np.clip((u*size).astype(int),0,size-1);iy=np.clip((v*size).astype(int),0,size-1)
        output[mask]=arrays[i][iy,ix]
    return Image.fromarray(output)


def prepare_cubemap(source,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    faces=split_legacy_cubemap(Path(source).read_bytes())
    for face in faces:
        (output/(face['name']+'.dds')).write_bytes(face['dds'])
        with Image.open(BytesIO(face['dds'])) as image:image.save(output/(face['name']+'.png'))
    cubemap_latlong(faces).save(output/'environment_latlong.png')
    return output/'environment_latlong.png'
