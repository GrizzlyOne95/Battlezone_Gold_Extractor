import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from test_scene import skeleton_blob,render_blob
from asura_scene import read_skeleton,read_render_mesh
from test_materials import table
from asura_materials import read_material_table


def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,TOOLS/filename);result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result);return result


class LibraryTests(unittest.TestCase):
    def test_animation_binding_uses_exact_names_and_numeric_duplicates(self):
        inspector=load('animation_inspector_test','inspect-asura-animations.py')
        for track in ('Railgun_Fire','Railgun_Fire.001','Railgun_Fire.023'):
            self.assertTrue(inspector.matches_clip(track,'Railgun_Fire'))
        for track in ('Railgun_Fire_Loop','Railgun_Fire.foo','Railgun_Fire.001.extra','Railgun_Fire.'):
            self.assertFalse(inspector.matches_clip(track,'Railgun_Fire'))

    def test_animation_audit_rejects_an_incomplete_quaternion_binding(self):
        from types import SimpleNamespace as NS
        inspector=load('animation_audit_test','inspect-asura-animations.py')
        class Owner(dict):pass
        owner=Owner(_asura_node_index=7);owner.name='root motion';owner.type='EMPTY'
        curves=[NS(data_path='rotation_quaternion',array_index=i) for i in range(4)]
        bag=NS(fcurves=curves)
        action=NS(layers=[NS(strips=[NS(channelbag=lambda slot:bag)])])
        strip=NS(action=action,action_slot=object())
        owner.animation_data=NS(nla_tracks=[NS(name='Deploy.001',strips=[strip])])
        document=dict(animations=[dict(name='Deploy',channels=[dict(target=dict(node=7,path='rotation'))])])
        self.assertTrue(inspector.audit_bindings(document,[owner])['passed'])
        curves.pop()
        result=inspector.audit_bindings(document,[owner])
        self.assertFalse(result['passed'])
        self.assertEqual(result['missing_channels'],[dict(clip='Deploy',node=7,path='rotation')])

    def test_requested_categories_preserve_variants_and_components(self):
        library=load('asset_library_test','build-asura-asset-library.py')
        cases=[('chars__actors_pc','Heavy_Tank_H0','enemy_tanks'),('chars__actors_pc','Missile_Drone_6','drones_and_enemies'),('chars__actors_pc','Toaster_Gun_Turret','turrets'),('envs__ice','IceRock01','props_and_structures'),('dlcdata__skin_example','Tank','player_tanks')]
        for ns,name,expected in cases:self.assertEqual(library.category(ns,name),expected)
        self.assertEqual(library.category('chars__actors_pc','Gatling_Gun'),'player_weapons')
        self.assertEqual(library.player_weapon_role('Weapon_Hardpoint00_Left'),'external_mount_and_doors')
        self.assertEqual(library.player_weapon_role('Guided_Missile_Launcher_Front_Ring'),'weapon_component')

    def test_native_bounds_include_blended_parent_transforms(self):
        library=load('asset_bounds_test','build-asura-asset-library.py');sk=read_skeleton(skeleton_blob());mesh=read_render_mesh(render_blob());mesh.joints[:,1]=1;mesh.weights[:,0]=.5;mesh.weights[:,1]=.5
        low,high=library.bounds(dict(skeleton=sk,mesh=mesh))
        np.testing.assert_allclose(low,[.5,1,1.5]);np.testing.assert_allclose(high,[.75,1.25,1.5])

    def test_gltf_carries_native_tint_model_and_namespace(self):
        exporter=load('tint_scene_test','export-asura-scenes.py');record=read_material_table(table(model=3))['materials'][123];scene=exporter.Scene(Path('unused.gltf'),{},dict(fixture={123:record}));index=scene.material(123,'fixture');material=scene.doc['materials'][index]
        self.assertEqual(material['extras']['native_namespace'],'fixture')
        self.assertEqual(material['pbrMetallicRoughness']['metallicFactor'],1)
        np.testing.assert_allclose(material['pbrMetallicRoughness']['baseColorFactor'],[12/255,16/255,37/255,1])

    def test_gltf_normal_preview_decodes_packed_rg(self):
        from PIL import Image
        exporter=load('normal_scene_test','export-asura-scenes.py');record=read_material_table(table())['materials'][123];record['texture_hashes']['normal']=789
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'normal.png';Image.new('RGBA',(1,1),(128,64,0,255)).save(source)
            scene=exporter.Scene(root/'asset.gltf',{789:[('fixture',source)]},dict(fixture={123:record}));material=scene.doc['materials'][scene.material(123,'fixture')]
            tex=scene.doc['textures'][material['normalTexture']['index']];derived=root/scene.doc['images'][tex['source']]['uri']
            with Image.open(derived) as image:r,g,b,a=image.getpixel((0,0))
            self.assertEqual((r,g,a),(128,191,255));self.assertGreater(b,230)
            self.assertIn('native_packed_normal_texture',material['extras'])


if __name__=='__main__':unittest.main()
