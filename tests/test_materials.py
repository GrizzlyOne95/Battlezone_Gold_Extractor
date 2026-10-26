import struct
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from asura_materials import read_material_table,surface_parameters


def table(model=0):
    record=bytearray(356);struct.pack_into('<3I',record,0,123,348,456)
    struct.pack_into('<I',record,0x40,model)
    struct.pack_into('<4f',record,0x48,40,0,.02353,1)
    struct.pack_into('<f',record,0xe4,.005)
    struct.pack_into('<3f',record,0x150,12/255,16/255,37/255)
    lut=struct.pack('<4f',.1,.2,.3,.4)*256
    return struct.pack('<4I',1,2,1,456)+lut+record


class MaterialTests(unittest.TestCase):
    def test_full_table_and_response_payload_survive(self):
        raw=table();decoded=read_material_table(raw)
        self.assertEqual(decoded['response_luts'][456]['payload'],raw[16:4112])
        self.assertEqual(decoded['materials'][123]['offset'],4112)
        self.assertEqual(decoded['materials'][123]['response_lut_hash'],456)
        self.assertEqual(decoded['materials'][123]['shader_hash'],456)

    def test_tint_emission_and_parallax_offsets(self):
        surface=read_material_table(table())['materials'][123]['surface_parameters']
        for actual,expected in zip(surface['albedo_modifier'],[12/255,16/255,37/255]):self.assertAlmostEqual(actual,expected)
        self.assertEqual(surface['self_illumination_scale'],40)
        self.assertAlmostEqual(surface['self_illumination_colour'][1],.02353)
        self.assertAlmostEqual(surface['parallax_bias'],-.0025)

    def test_only_native_model_three_is_metal(self):
        for model in (0,1,3):
            surface=read_material_table(table(model))['materials'][123]['surface_parameters']
            self.assertEqual(surface['metallic'],1 if model==3 else 0)

    def test_bad_header_and_truncated_lut_rejected(self):
        with self.assertRaisesRegex(ValueError,'size'):read_material_table(table()[:-1])
        raw=bytearray(table());struct.pack_into('<I',raw,4,99)
        with self.assertRaisesRegex(ValueError,'version'):read_material_table(raw)

    def test_missing_response_reference_rejected(self):
        raw=bytearray(table());struct.pack_into('<I',raw,4112+8,999)
        with self.assertRaisesRegex(ValueError,'response LUT'):read_material_table(raw)

    def test_nonfinite_named_scalar_is_not_silently_used(self):
        raw=bytearray(table());struct.pack_into('<f',raw,4112+0x150,float('nan'))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):read_material_table(raw)


if __name__=='__main__':unittest.main()
