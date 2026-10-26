import importlib.util
from pathlib import Path
import struct
import sys
import unittest

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
spec=importlib.util.spec_from_file_location('weapon_definition_test',TOOLS/'export-gold-weapon-definitions.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class GoldWeaponDefinitionTests(unittest.TestCase):
    def test_controller_links_require_both_known_model_and_clip(self):
        data=struct.pack('<5I',1,1004,8,123,456)+b'\xff'*71
        result=module.controller_record(data,{123:'Autocannon'},{456:dict(name='Autocannon_Fire',duration=.2)})
        self.assertEqual(result['clip'],'Autocannon_Fire')
        self.assertEqual(result['clip_duration'],.2)
        self.assertNotIn('damage',result)
        self.assertNotIn('fire_rate',result)
        self.assertIsNone(module.controller_record(data,{123:'Autocannon'},{}))
        self.assertIsNone(module.controller_record(data,{}, {456:dict(name='Fire',duration=.2)}))

    def test_unknown_controller_layout_is_not_given_known_semantics(self):
        self.assertIsNone(module.controller_record(struct.pack('<5I',2,1004,8,123,456),{123:'Gun'},{456:dict(name='Fire',duration=.2)}))
        self.assertIsNone(module.controller_record(b'\0'*12,{},{}))


if __name__=='__main__':unittest.main()
