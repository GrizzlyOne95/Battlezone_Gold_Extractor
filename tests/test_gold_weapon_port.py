from pathlib import Path
import sys
import unittest
import importlib.util

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from gold_weapons import parse_odf, serialize_odf, verify_render_graph, recipes, render_sections, explosion_sections
try:
    from lupa import LuaRuntime
except ImportError:
    LuaRuntime=None


class WeaponPortTests(unittest.TestCase):
    def test_all_families_have_resolving_effect_graphs_and_one_muzzle_emitter(self):
        for row in recipes():
            stem='bo'+row['id'];xpl='bx'+row['id']
            docs={stem:parse_odf(serialize_odf(render_sections(stem,(100,210,255)))),xpl:parse_odf(serialize_odf(explosion_sections(xpl,row,(100,210,255))))}
            roots=[stem+'.Render']+[xpl+'.'+name for name in ('Core','Smoke','Spark')]
            reached=verify_render_graph(docs,roots)
            self.assertEqual(len(reached),8)
            self.assertGreater(float(docs[stem]['flash']['emitdelay']),1000)
            self.assertEqual(float(docs[xpl]['explosionclass']['damageconcussion']),row['blast_damage'])

    def test_render_graph_rejects_missing_children_counts_and_cycles(self):
        cases=[{'renderbase':'draw_multi','rendercount':'2','rendername1':'p.child'}, {'renderbase':'draw_multi','rendercount':'1','rendername1':'p.absent'}, {'renderbase':'draw_emit','emitname':'p.root'}]
        for root in cases:
            with self.assertRaises(ValueError):verify_render_graph({'p':{'root':root,'child':{'renderbase':'draw_twirl'}}},['p.root'])

    def test_odf_parser_retains_qualified_names_and_rejects_shadowed_settings(self):
        doc=parse_odf('[Render] // note\nrenderName = "boauto.Muzzle" // child\n')
        self.assertEqual(doc['render']['rendername'],'boauto.Muzzle')
        for text in ('[A]\nx=1\nx=2','[A]\nx=1\n[A]\ny=2'):
            with self.assertRaises(ValueError):parse_odf(text)

    @unittest.skipIf(LuaRuntime is None,'Optional Lua runtime is unavailable')
    def test_lua_helper_loads_without_side_effects_and_routes_explicit_requests(self):
        path=Path(__file__).resolve().parents[1]/'tools/build-gold-weapons-mod.py'
        spec=importlib.util.spec_from_file_location('weapon_builder_test',path);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        lua=LuaRuntime(unpack_returned_tuples=True);calls=[]
        lua.globals().GiveWeapon=lambda *args:calls.append(args)
        lua.globals().BuildObject=lambda *args:('pickup',*args)
        rows=[dict(id=r['id'],weapon='bg'+r['id'],pickup='ba'+r['id']) for r in recipes()]
        helper=lua.execute(builder.lua_helper(rows));self.assertEqual(calls,[])
        helper.Equip(42,'rail',2);self.assertEqual(calls,[(42,'bgrail',2)])
        self.assertEqual(helper.Pickup('guid',1,99),('pickup','baguid',1,99))
        for id_,slot in [('unknown',0),('main',-1),('main',1.5),('main',5)]:
            with self.assertRaises(Exception):helper.Equip(42,id_,slot)
        self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
