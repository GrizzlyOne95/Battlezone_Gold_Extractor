"""MARE tables and surface constants traced through the native DX11 code.

Offsets are serialized record offsets, including hash/length. Shader_hash was
an early name for the response-LUT hash; retain it as a compatibility alias.
"""
from __future__ import annotations

import hashlib
import math
import struct


def surface_parameters(words):
    if len(words)!=89:raise ValueError('Expected 356-byte material record')
    def scalar(offset):
        value=struct.unpack('<f',struct.pack('<I',words[offset//4]))[0]
        if not math.isfinite(value):raise ValueError(f'Nonfinite material scalar at {offset:#x}')
        return value
    model=words[0x40//4]
    return dict(albedo_modifier=[scalar(0x150+i*4) for i in range(3)],
                material_model=model,metallic=1.0 if model==3 else 0.0,
                self_illumination_scale=scalar(0x48),
                self_illumination_colour=[scalar(0x4c+i*4) for i in range(3)],
                reflectivity=scalar(0xdc),refractivity=scalar(0x134),
                parallax_scale=scalar(0xe4),parallax_bias=-0.5*scalar(0xe4),
                native_feature_bits=f'{words[0x130//4]:08x}{words[0x12c//4]:08x}',
                runtime_overrides='Entity callbacks can replace albedo/emission constants; these are serialized defaults.')


def read_material_table(blob):
    if len(blob)<12:raise ValueError('Truncated MARE header')
    count,version,lut_count=struct.unpack_from('<3I',blob)
    if version!=2:raise ValueError('Unsupported MARE table version')
    if count>1_000_000 or lut_count>65536:raise ValueError('Invalid MARE counts')
    start=12+lut_count*4100
    if start+count*356!=len(blob):raise ValueError('MARE counts do not match exact table size')
    luts={}
    for index in range(lut_count):
        offset=12+index*4100;key=struct.unpack_from('<I',blob,offset)[0]
        payload=blob[offset+4:offset+4100]
        if key in luts:raise ValueError('Duplicate MARE response LUT hash')
        luts[key]=dict(hash=key,index=index,offset=offset+4,payload=payload,sha256=hashlib.sha256(payload).hexdigest())
    materials={}
    for index in range(count):
        offset=start+index*356;words=struct.unpack_from('<89I',blob,offset)
        if words[1]!=348:raise ValueError('Unsupported material stride')
        if words[0] in materials:raise ValueError('Duplicate MARE material hash')
        if words[2] not in luts:raise ValueError('Material response LUT is absent from its table')
        materials[words[0]]=dict(hash=words[0],offset=offset,response_lut_hash=words[2],shader_hash=words[2],
            texture_hashes={role:words[at//4] for role,at in [('base_color',12),('normal',16),('specular',20),('opacity',24),('emissive',28),('reflection',68)]},
            raw_words=list(words),surface_parameters=surface_parameters(words))
    return dict(version=version,response_luts=luts,materials=materials)
