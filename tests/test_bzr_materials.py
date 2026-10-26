import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from bzr_materials import with_stock_import, validate_stock_inheritance


class MaterialImportTests(unittest.TestCase):
    STOCK='abstract pass BZPassSchemeHigh {}\nmaterial BZBase {}\nmaterial BZBaseCockpit {}'
    GENERATED='material Gold/Masked { technique { pass : BZPassSchemeHigh {} } }\nmaterial Gold/Hull : BZBase {}\nmaterial Gold/Glass : Gold/Masked {}\nmaterial Gold/Cabin : BZBaseCockpit {}'

    def test_exporter_failure_requires_explicit_stock_import(self):
        with self.assertRaisesRegex(ValueError,'Missing explicit'):validate_stock_inheritance(self.GENERATED,self.STOCK)
        result=validate_stock_inheritance(with_stock_import(self.GENERATED),self.STOCK)
        self.assertEqual(result['stock_parents'],['BZBase','BZBaseCockpit','BZPassSchemeHigh'])

    def test_import_fix_is_idempotent_and_unknown_parents_remain_errors(self):
        fixed=with_stock_import(self.GENERATED)
        self.assertEqual(with_stock_import(fixed),fixed)
        with self.assertRaisesRegex(ValueError,'Unresolved'):validate_stock_inheritance(fixed.replace('BZPassSchemeHigh','MissingPass'),self.STOCK)


if __name__=='__main__':unittest.main()
