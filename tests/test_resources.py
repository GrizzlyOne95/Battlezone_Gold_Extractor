"""Bounds, payload preservation and native channel regression cases."""
import importlib.util
from pathlib import Path
import struct
import sys
import unittest

import numpy as np

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from asura_shaders import read_dxbc,read_shader_bank
from asura_textures import split_legacy_cubemap,cubemap_latlong
from asura_scene import read_render_mesh,read_skeleton
from test_scene import render_blob,skeleton_blob,name


def dxbc():
    program=struct.pack('<II',0x50,2)
    return b'DXBC'+bytes(16)+struct.pack('<IIII',1,52,1,36)+b'SHDR'+struct.pack('<I',len(program))+program


def cube():
    header=bytearray(128);header[:4]=b'DDS '
    struct.pack_into('<I',header,4,124)
    struct.pack_into('<6I',header,8,0x100f,2,2,8,0,1)
    struct.pack_into('<8I',header,76,32,0x41,0,32,0xff,0xff00,0xff0000,0xff000000)
    struct.pack_into('<II',header,108,0x1008,0xfe00)
    colors=[(255,0,0,255),(0,255,0,255),(0,0,255,255),(255,255,0,255),(255,0,255,255),(0,255,255,255)]
    return bytes(header)+b''.join(bytes(c)*4 for c in colors),colors


class ResourceTests(unittest.TestCase):
    def test_dxbc_bounds_and_shader_profile(self):
        info=read_dxbc(dxbc());self.assertEqual(info['profile'],'ps_5_0');self.assertEqual(info['size'],52)
        bad=bytearray(dxbc());struct.pack_into('<I',bad,32,100)
        with self.assertRaisesRegex(ValueError,'offset'):read_dxbc(bad)
        with self.assertRaisesRegex(ValueError,'bounds'):read_dxbc(dxbc()[:-1])

    def test_shader_bank_preserves_bytecode_and_permutation(self):
        raw=dxbc();bank=struct.pack('<IQI',1,0x123456789abcdef0,len(raw))+raw
        entries,tail=read_shader_bank(bank)
        self.assertEqual(entries[0].bytecode,raw);self.assertEqual(entries[0].permutation,0x123456789abcdef0);self.assertEqual(tail,b'')
        with self.assertRaisesRegex(ValueError,'tail'):read_shader_bank(bank+b'\0')

    def test_particle_bank_retains_unknown_tail(self):
        raw=dxbc();bank=struct.pack('<II',1,len(raw))+raw+b'abcd'
        entries,tail=read_shader_bank(bank,particle=True)
        self.assertEqual(entries[0].bytecode,raw);self.assertEqual(tail,b'abcd')
        with self.assertRaisesRegex(ValueError,'tail'):read_shader_bank(bank+b'\0',particle=True)

    def test_cubemap_keeps_all_six_native_payloads(self):
        raw,colors=cube();faces=split_legacy_cubemap(raw)
        self.assertEqual([f['name'] for f in faces],['px','nx','py','ny','pz','nz'])
        for f,color in zip(faces,colors):
            self.assertEqual(f['dds'][128:],bytes(color)*4)
            self.assertEqual(struct.unpack_from('<I',f['dds'],112)[0],0)
        with self.assertRaisesRegex(ValueError,'payload size'):split_legacy_cubemap(raw[:-1])

    def test_cubemap_projection_face_directions(self):
        raw,colors=cube();image=np.array(cubemap_latlong(split_legacy_cubemap(raw),width=16))
        np.testing.assert_array_equal(image[4,8],colors[4])
        np.testing.assert_array_equal(image[4,12],colors[0])
        np.testing.assert_array_equal(image[4,4],colors[1])
        np.testing.assert_array_equal(image[0,8],colors[2])

    def test_bone_palette_is_metadata_not_joint_remapping(self):
        raw=bytearray(render_blob());struct.pack_into('<I',raw,16,1)
        raw+=struct.pack('<3I',2,1,0)
        mesh=read_render_mesh(raw)
        self.assertEqual(mesh.bone_palettes,[[1,0]])
        np.testing.assert_array_equal(mesh.joints[:,0],0)
        with self.assertRaisesRegex(ValueError,'palette'):read_render_mesh(raw[:-1])

    def test_nonfinite_uv_retains_source_half_float_bits(self):
        raw=bytearray(render_blob());struct.pack_into('<H',raw,68+24,0x7c00)
        mesh=read_render_mesh(raw)
        self.assertEqual(mesh.uv0[0,0],0)
        self.assertEqual(mesh.nonfinite_uv,[dict(channel=0,vertex=0,component=0,native_f16_bits='7c00',interchange_value=0.)])

    def test_node_flag_one_survives_and_unknown_flag_rejected(self):
        raw=bytearray(skeleton_blob());offset=raw.index(name('Hull'))+len(name('Hull'));raw[offset]=1
        self.assertEqual(read_skeleton(raw).nodes[0].native_flags,1)
        raw[offset]=2
        with self.assertRaisesRegex(ValueError,'flag'):read_skeleton(raw)

    def test_volume_exact_bounds(self):
        spec=importlib.util.spec_from_file_location('resource_export',TOOLS/'export-asura-resources.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        payload=bytes(range(32));raw=struct.pack('<5I',1,123,2,2,2)+payload
        record=module.read_volumes(raw)[0];self.assertEqual(record['payload'],payload);self.assertEqual(record['size'],32)
        with self.assertRaisesRegex(ValueError,'bounds'):module.read_volumes(raw[:-1])
        with self.assertRaisesRegex(ValueError,'tail'):module.read_volumes(raw+b'\0')


if __name__=='__main__':unittest.main()
