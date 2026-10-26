import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


spec=importlib.util.spec_from_file_location('bzr_port_preparation',Path(__file__).resolve().parents[1]/'tools/prepare-bzr-port.py')
porter=importlib.util.module_from_spec(spec);spec.loader.exec_module(porter)


class BzrPortPreparationTests(unittest.TestCase):
    def test_duplicate_joint_slots_do_not_create_a_false_influence_warning(self):
        joints=np.array([[1,1,2,3],[1,2,3,4]])
        weights=np.array([[.2,.3,.4,.1],[.25,.25,.25,.25]])
        result=porter.skin_risks(joints,weights)
        self.assertEqual(result['vertices_above_three'],1)
        self.assertEqual(result['max_positive_influences'],4)
        self.assertEqual(result['max_discarded_weight_if_top3'],.25)
        np.testing.assert_array_equal(weights,[[.2,.3,.4,.1],[.25,.25,.25,.25]])

    def test_visual_aliases_are_unique_ascii_and_keep_lod_character_explicit(self):
        names=[porter.alias(i) for i in range(1000)]
        self.assertEqual(len(set(names)),1000)
        self.assertTrue(all(len(name.encode('ascii'))==8 and name[3]=='1' for name in names))
        self.assertEqual(porter.alias(17,cockpit=True)[3],'2')
        with self.assertRaises(ValueError):porter.alias(36**3)

    def test_staging_keeps_only_the_complete_hierarchy_and_rebases_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.gltf';out=root/'staging';out.mkdir();destination=out/'tank.gltf'
            document=dict(nodes=[dict(name='outside'),dict(name='Gallery',translation=[9,0,0],rotation=[1,0,0,0],children=[2,3]),dict(name='joint'),dict(name='mesh',mesh=0,skin=0)],
                meshes=[dict(primitives=[dict(material=1)])],materials=[dict(name='unused'),dict(name='used')],
                skins=[dict(joints=[2],skeleton=2)],animations=[dict(name='Fire',channels=[dict(sampler=0,target=dict(node=2,path='rotation')),dict(sampler=1,target=dict(node=0,path='translation'))],samplers=[])],
                scenes=[dict(nodes=[0,1])],buffers=[dict(uri='source.bin')],images=[dict(uri='paint%20blue.png')])
            source.write_text(json.dumps(document));original=source.read_bytes()
            receipt=porter.stage_assembly(source,1,destination);staged=json.loads(destination.read_text())
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(receipt['retained_nodes'],3)
            self.assertNotIn('translation',staged['nodes'][0])
            self.assertEqual(staged['nodes'][0]['rotation'],[1,0,0,0])
            self.assertEqual(staged['nodes'][0]['children'],[1,2])
            self.assertEqual(staged['skins'][0]['joints'],[1])
            self.assertEqual(staged['animations'][0]['channels'],[dict(sampler=0,target=dict(node=1,path='rotation'))])
            self.assertEqual(staged['materials'],[dict(name='used')])
            self.assertEqual(staged['meshes'][0]['primitives'][0]['material'],0)
            self.assertEqual(staged['images'][0]['uri'],'../paint%20blue.png')


if __name__=='__main__':unittest.main()
