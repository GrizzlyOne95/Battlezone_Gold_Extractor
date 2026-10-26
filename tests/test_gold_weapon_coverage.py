import struct
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from gold_weapons import read_gold_text, recipes
from gold_weapon_geometry import gltf_worlds, orientation_landmarks


class WeaponCoverageTests(unittest.TestCase):
    def text_fixture(self):
        names=['WEAPON_DESC_CANNON','WEAPON_DESC_LASER'];values=['Cannon description','Laser description'];records=b''
        for name,value in zip(names,values):
            hash_=0
            for char in name.lower().encode():hash_=(hash_*31+char)&0xffffffff
            encoded=(value+'\0').encode('utf-16le');records+=struct.pack('<2I',hash_,len(encoded)//2)+encoded
        keys=b''.join(n.encode()+b'\0' for n in names)
        return struct.pack('<4I',2,0,len(records)-16,0)+records+b'Test\0\0\0\0'+struct.pack('<I',len(keys))+keys

    def test_native_text_pairing_checks_lengths_and_hashes(self):
        blob=self.text_fixture();self.assertEqual(read_gold_text(blob)['WEAPON_DESC_LASER'],'Laser description')
        for bad in (blob[:-1],blob.replace(b'WEAPON_DESC_LASER',b'WEAPON_DESC_OTHER'),blob[:16]+b'\0\0\0\0'+blob[20:]):
            with self.assertRaises(ValueError):read_gold_text(bad)

    def test_semantic_weapon_variants_are_present_and_ids_fit_legacy_names(self):
        rows=recipes();self.assertEqual(len({r['id'] for r in rows}),len(rows))
        for row in rows:self.assertLessEqual(len('bo'+row['id']+'f'),8)
        kinds={r['kind'] for r in rows}
        self.assertTrue({'beam','charge_beam','flame','remote','cluster','burst','time_fuse','barrage','flare'}<=kinds)

    def test_named_front_rear_landmarks_survive_parent_rotation(self):
        doc={'nodes':[{'name':'root','rotation':[1,0,0,0],'children':[1,2]}, {'name':'Missile_frontwing_01','translation':[0,0,1]}, {'name':'Missile_rearwing_01','translation':[0,0,-1]}]}
        forward=orientation_landmarks(doc)[0]['forward'];np.testing.assert_allclose(forward,[0,0,1])
        doc['nodes'][0]['rotation']=[0,0,0,1]
        backward=orientation_landmarks(doc)[0]['forward'];self.assertLess(backward[2],0)

    def test_column_major_socket_matrix_composes_with_parent(self):
        matrix=np.eye(4);matrix[:3,3]=[1,2,3]
        doc={'nodes':[{'name':'root','translation':[5,0,0],'children':[1]},{'name':'socket:Thrust','matrix':matrix.T.reshape(-1).tolist()}]}
        np.testing.assert_allclose(gltf_worlds(doc)[1][:3,3],[6,2,3])


if __name__=='__main__':unittest.main()
