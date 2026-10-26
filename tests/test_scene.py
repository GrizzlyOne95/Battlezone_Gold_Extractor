import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from asura_scene import (name_hash, sample_rotations, read_skeleton, read_render_mesh,
                         read_animation, read_markers, validate_render_binding,coalesce_skin_influences)


def name(value):
    value = value.encode() + b'\0'
    return value + bytes((-len(value)) % 4)


def skeleton_blob(parents=(0, 0)):
    points = np.array([[0, 0, 0], [0.25, 0, 0], [0, 0.25, 0]], dtype='<f4')
    bounds = [0, .25, 0, .25, 0, 0]
    geometry = struct.pack('<8I', 14, 3, 1, 0, 1, 1, 0xffff0000, 0)
    geometry += struct.pack('<7f', *bounds, .4) + points.tobytes() + struct.pack('<4H', 0, 1, 2, 65535)
    blob = struct.pack('<II', 0, 2) + name('Tank') + struct.pack('<2I', *parents)
    blob += struct.pack('<14f', 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 0, 0, 0, 1) + b'\x01'
    blob += name('Hull') + b'\0' + struct.pack('<I', len(geometry)) + geometry
    blob += name('Gun') + b'\0' + struct.pack('<I', 0)
    return blob


def render_blob():
    header = struct.pack('<4I', 1, 3, 3, 1) + struct.pack('<6I', 0, 123, 0, 3, 0, 1) + struct.pack('<I', 0) + struct.pack('<6f', 1, 1, 1, 0, 0, 0)
    vertices = bytearray()
    for i, pos in enumerate([(0, 0, 0), (16384, 0, 0), (0, 16384, 0)]):
        v = bytearray(40)
        struct.pack_into('<4H', v, 0, *pos, 65535)
        struct.pack_into('<I', v, 8, 512 | (512 << 10) | (1023 << 20))
        struct.pack_into('<2e', v, 24, i / 2, 1 - i / 2)
        v[32] = 255
        vertices.extend(v)
    return header + vertices + struct.pack('<3H', 0, 1, 2)


def animation_blob(root_motion=False):
    nq = nt = 2 if root_motion else 1
    blob = struct.pack('<IfIIII', 1, 1, 0, nq, nt, 0) + name('Gun_Fire')
    blob += struct.pack('<HHII', 1, 1, 0, 0)
    if root_motion:
        blob += struct.pack('<HHII', 1, 1, 1, 1)
    blob += struct.pack('<4H', 32767, 32767, 32767, 65535) * nq
    blob += struct.pack('<' + 'H' * nq, *([0] * nq))
    blob += struct.pack('<3f', 1, 2, 3) * nt
    blob += struct.pack('<' + 'H' * nt, *([0] * nt))
    blob += bytes(16) + struct.pack('<III', name_hash('Gun'), 1, 0)
    return blob


class SceneTests(unittest.TestCase):
    def test_repeated_joint_weights_coalesce_without_deformation_change(self):
        joints=np.array([[13,0,14,0],[13,14,0,0]],dtype=np.uint8)
        weights=np.array([[249,5,0,1],[252,1,1,1]],dtype=np.float32)/255
        original=weights.copy();js,ws,records=coalesce_skin_influences(joints,weights)
        transforms=np.arange(15*3,dtype=float).reshape(15,3)
        np.testing.assert_allclose((transforms[js]*ws[:,:,None]).sum(1),(transforms[joints]*weights[:,:,None]).sum(1),atol=1e-6)
        np.testing.assert_array_equal(weights,original)
        self.assertEqual(ws[0,3],0);self.assertEqual(ws[1,3],0)
        self.assertEqual(records[0]['native_weight_bytes'],[249,5,0,1])

    def test_root_motion_does_not_animate_fixed_attachment_calibration(self):
        spec=importlib.util.spec_from_file_location('calibrated_scene_export',TOOLS/'export-asura-scenes.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        asset=dict(skeleton=read_skeleton(skeleton_blob()),mesh=read_render_mesh(render_blob()),namespace='test',source='fixture',animations=[read_animation(animation_blob(True))],markers=[])
        scene=module.Scene(Path('unused.gltf'),{},{});wrapper,joints,_=scene.add_asset(asset,transform=np.diag([1/32,1/32,1/32,1]))
        pivots=[i for i,n in enumerate(scene.doc['nodes']) if n.get('extras',{}).get('native_root_motion_pivot')]
        self.assertEqual(len(pivots),1);self.assertIn(pivots[0],scene.doc['nodes'][wrapper]['children']);self.assertIn(joints[0],scene.doc['nodes'][pivots[0]]['children'])
        self.assertTrue(all('matrix' not in scene.doc['nodes'][c['target']['node']] for a in scene.doc['animations'] for c in a['channels']))

    def test_native_name_hash_matches_material_and_node_ids(self):
        self.assertEqual(name_hash('Tank_Prototype_01_a'), 0xd5929695)
        self.assertEqual(name_hash('Tank_Prototype_01_a_Body'), 0xd046310c)

    def test_hierarchy_and_local_transforms_survive(self):
        sk = read_skeleton(skeleton_blob())
        self.assertEqual([(n.name, n.parent) for n in sk.nodes], [('Hull', -1), ('Gun', 0)])
        np.testing.assert_array_equal(sk.nodes[1].translation, [1, 2, 3])
        np.testing.assert_array_equal(sk.nodes[0].cpu_positions, [[0, 0, 0], [.25, 0, 0], [0, .25, 0]])
        np.testing.assert_array_equal(sk.nodes[0].cpu_faces, [[0, 1, 2, 65535]])

    def test_cyclic_hierarchy_rejected(self):
        with self.assertRaisesRegex(ValueError, 'parent'):
            read_skeleton(skeleton_blob((0, 1)))

    def test_render_layout_uvs_and_skinning(self):
        mesh = read_render_mesh(render_blob())
        np.testing.assert_allclose(mesh.positions, [[0, 0, 0], [.25, 0, 0], [0, .25, 0]])
        np.testing.assert_allclose(mesh.uv0, [[0, 1], [.5, .5], [1, 0]])
        np.testing.assert_array_equal(mesh.triangles, [[0, 1, 2]])
        self.assertEqual(mesh.parts[0]['material_hash'], 123)
        validation = validate_render_binding(mesh, read_skeleton(skeleton_blob()))
        self.assertTrue(all(x['within_quantization'] for x in validation['bounds_checks']))

    def test_render_layout_does_not_move_header_bytes_into_vertices(self):
        mesh = read_render_mesh(render_blob())
        self.assertEqual(mesh.positions[0].tolist(), [0, 0, 0])
        bad = bytearray(render_blob())
        struct.pack_into('<H', bad, len(bad) - 2, 999)
        with self.assertRaisesRegex(ValueError, 'index'):
            read_render_mesh(bytes(bad))

    def test_animation_keys_times_and_node_binding(self):
        animation = read_animation(animation_blob())
        self.assertEqual(animation.node_hashes, (name_hash('Gun'),))
        np.testing.assert_allclose(animation.rotations, [[0, 0, 0, 1]])
        np.testing.assert_array_equal(animation.translations, [[1, 2, 3]])
        np.testing.assert_array_equal(animation.rotation_times, [0])
        self.assertIsNone(animation.root_motion_track)

    def test_separate_root_motion_track_is_not_lost(self):
        animation = read_animation(animation_blob(True))
        self.assertEqual(animation.root_motion_track, (1, 1, 1, 1))
        self.assertEqual(len(animation.rotations), 2)

    def test_rotation_sampling_keeps_native_keys_and_unit_quaternions(self):
        times = np.array([0, .233333, 1],dtype=np.float32)
        values = np.array([[0,0,0,1],[.5,0,0,np.sqrt(.75)],[0,0,0,1]],dtype=np.float32)
        sampled_times, sampled_values = sample_rotations(times,values)
        self.assertTrue(np.all(np.isin(times,sampled_times)))
        np.testing.assert_allclose(np.linalg.norm(sampled_values,axis=1),1,atol=1e-6)
        for time, value in zip(times,values):
            np.testing.assert_allclose(sampled_values[np.flatnonzero(sampled_times==time)[0]],value,atol=1e-6)

    def test_markers_retain_native_parent_and_basis(self):
        blob = struct.pack('<I', 1) + name('Tank')
        blob += struct.pack('<12fI', 1, 2, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1) + name('Muzzle0')
        owner, markers = read_markers(blob)
        self.assertEqual(owner, 'Tank')
        self.assertEqual(markers[0]['name'], 'Muzzle0')
        self.assertEqual(markers[0]['translation'], [1, 2, 3])
        self.assertEqual(markers[0]['parent'], 1)

    def test_gltf_retains_hierarchy_materials_uvs_and_animation(self):
        spec = importlib.util.spec_from_file_location('scene_export', TOOLS / 'export-asura-scenes.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tank.gltf'
            scene = module.Scene(path, {}, {})
            asset = dict(skeleton=read_skeleton(skeleton_blob()), mesh=read_render_mesh(render_blob()), namespace='test', source='fixture', animations=[read_animation(animation_blob())], markers=[])
            scene.add_asset(asset)
            scene.save()
            doc = json.loads(path.read_text())
            self.assertEqual(doc['nodes'][1]['children'], [2])
            self.assertEqual(doc['nodes'][2]['translation'], [1, 2, 3])
            self.assertIn('TEXCOORD_0', doc['meshes'][0]['primitives'][0]['attributes'])
            self.assertIn('WEIGHTS_0', doc['meshes'][0]['primitives'][0]['attributes'])
            self.assertEqual(doc['animations'][0]['channels'][0]['target']['node'], 2)
            self.assertEqual(doc['materials'][0]['extras']['native_material_hash'], '0000007b')
            self.assertTrue(path.with_suffix('.bin').exists())

    def test_legacy_obj_backends_use_correct_native_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for module_name, filename in [('legacy_hskn', 'export-hskn-obj-candidates.py'), ('legacy_render', 'export-model-obj-candidates.py')]:
                spec = importlib.util.spec_from_file_location(module_name, TOOLS / filename)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                if module_name == 'legacy_hskn':
                    path = root / 'HSKN_chunk' / '.1.dat'
                    path.parent.mkdir()
                    path.write_bytes(skeleton_blob())
                    result, error = module.parse_hskn(path, root)
                    self.assertIsNone(error)
                    np.testing.assert_allclose(result.vertices, [[0, 0, 0], [.25, 0, 0], [0, .25, 0]])
                    np.testing.assert_array_equal(result.triangles, [[0, 1, 2]])
                else:
                    path = root / 'tank'
                    path.write_bytes(render_blob())
                    result, error = module.parse_model_file(path, root)
                    self.assertIsNone(error)
                    np.testing.assert_allclose(module.decode_native(result), [[0, 0, 0], [.25, 0, 0], [0, .25, 0]])


if __name__ == '__main__':
    unittest.main()
