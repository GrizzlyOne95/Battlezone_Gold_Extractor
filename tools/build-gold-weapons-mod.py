"""Build Gold player weapon profiles, native tier names and BZR presentation assets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile
import csv
import copy

from PIL import Image, ImageDraw
from gold_weapons import recipes, parse_odf, serialize_odf, render_sections, explosion_sections, q, read_gold_text
from bzr_materials import with_stock_import

REPO=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def template(path):
    return {s:{k:(v if re.fullmatch(r'[-+0-9.eEfF]+',v) else q(v)) for k,v in values.items()} for s,values in parse_odf(path.read_text()).items()}


def audio(base,out,ffmpeg,recipe,impact):
    token=recipe['impact_token' if impact else 'fire_token']
    if token is None:return None
    candidates=[]
    for ns in ('sounds__gmsnd_asr','chars__actors_pc'):
        root=base/'audio_pcm'/ns/'sounds'
        group=root/('weapons/pc/emp' if impact and token=='/emp/' else 'impacts/projectiles' if impact else 'weapons')
        candidates.extend(p for p in group.rglob('*.wav') if (token in '/'+p.relative_to(root).as_posix().lower()) and not any(t in p.relative_to(root).parts for t in ('npc','rpc')))
    candidates=sorted(candidates,key=lambda p:('sounds__gmsnd_asr' not in str(p),str(p)))
    if impact and token=='/emp/':candidates=[p for p in candidates if 'impact_blast' in p.name.lower()]
    if not impact:candidates=[p for p in candidates if any(t in p.name.lower() for t in ('blast','final','fire','loop')) and not any(t in p.name.lower() for t in ('tail','reload','dry_fire','unstow','stow','heat_up','cool_down'))]
    if not impact and recipe['id']=='chrg':candidates=sorted(candidates,key=lambda p:('fire_stop' not in p.name,str(p)))
    if not candidates:raise ValueError('No audio candidate for '+recipe['family']+(' impact' if impact else ' fire'))
    # Stable presentation selection, not a claim of a decoded runtime sound event.
    source=candidates[0];dest=out/('bg'+recipe['id']+('i' if impact else 'f')+'.wav')
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-y','-i',str(source)]
    processing='Mono PCM16 22050 Hz; original duration retained'
    if not impact:
        length=min(2.,max(recipe['delay']*.9,.12));fade=min(.03,length/4)
        cmd+=['-t',str(length),'-af',f'afade=t=in:st=0:d=0.003,afade=t=out:st={length-fade}:d={fade}']
        processing=f'Mono PCM16 22050 Hz; cropped to at most {length:.3f}s with edge fades for authored BZR cadence'
    cmd+=['-ac','1','-ar','22050','-c:a','pcm_s16le',str(dest)]
    subprocess.run(cmd,check=True,capture_output=True)
    return dict(file=dest.name,source=str(source),source_sha256=digest(source),candidates=[str(p) for p in candidates],selection='Filename/category candidate; native event routing not decoded',processing=processing,
        shared_source_fallback=recipe['id'] in ('multi','laser','80','flak','fire','split','cemp','amag','brst','barr'))


def audio_cues(base,out,ffmpeg,recipe,fire):
    """Keep native lifecycle cues; wire only events BZR actually exposes here."""
    token=recipe['fire_token'];matches=[]
    for ns in ('sounds__gmsnd_asr','chars__actors_pc'):
        root=base/'audio_pcm'/ns/'sounds/weapons'
        matches.extend(p for p in root.rglob('*.wav') if token in '/'+p.relative_to(root).as_posix().lower() and not any(x in p.parts for x in ('npc','rpc')))
    result=[]
    for index,(cue,terms) in enumerate([('tail',('tail',)),('loop',('fire_loop',)),('start',('fire_start','heat_up')),('stop',('fire_stop',)),('reload',('reload',)),('stow',('_stow',)),('unstow',('_unstow',)),('dry',('dry_fire',))]):
        candidates=[p for p in matches if any(t in p.name.lower() for t in terms)]
        # Reload/stow/dry cues are often in a shared bank rather than gun's folder.
        if not candidates and cue in ('reload','stow','unstow','dry'):
            aliases={'main':'cannon','auto':'autocannon','gatl':'machinegun','pist':'pistol','arty':'artillery','rock':'rocket','guid':'guided_missile','multi':'multi_missile','laser':'laser_missile','rail':'railgun'}
            name=aliases.get(recipe['id'])
            if name:
                for ns in ('sounds__gmsnd_asr','chars__actors_pc'):
                    candidates.extend(p for p in (base/'audio_pcm'/ns/'sounds/weapons/pc').rglob('*.wav') if name in p.name.lower() and any(t in p.name.lower() for t in terms))
        if not candidates:continue
        source=sorted(candidates,key=lambda p:('sounds__gmsnd_asr' not in str(p),str(p)))[0];dest=out/('c'+recipe['id'][:4]+str(index)+'.wav')
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-y','-i',str(source),'-ac','1','-ar','22050','-c:a','pcm_s16le',str(dest)],check=True,capture_output=True)
        result.append(dict(cue=cue,file=dest.name,source=str(source),source_sha256=digest(source),runtime_connection='unassigned; retained native cue for later event/controller work',processing='Full duration, mono PCM16 22050 Hz'))
    return result


def customize(recipe,weapon,ord_doc,xpl_doc,ord_,xpl,fire,cues):
    """Use existing BZR classes for mechanics with explicit fidelity limits."""
    kind=recipe['kind'];id_=recipe['id'];extra={};limits=[]
    if kind in ('beam','charge_beam'):
        ord_doc['ordnanceclass'].update(lifespan=180/recipe['speed'],damageballistic=0,damageflame=recipe['damage'])
        ord_doc['beamclass']={}
        ord_doc['Core'].update(renderBase=q('draw_bolt'),textureName=q('bgbeam.tga'),segmentLength=3,segmentVariance=q('0.0 0.0 0.0'),textureRate=.08,textureSpeed=-2,animateTime=0,startRadius=.18,finishRadius=.18)
        limits.append('Native BZR beam; no recovered Gold heat controller')
    if kind in ('charge_beam','burst'):
        weapon.pop('cannonclass',None);settings={'startRate':11025.,'deltaRate':1500.,'startVolume':40.,'deltaVolume':10.,'ordnanceCount':3 if kind=='charge_beam' else 2}
        charge=next((c for c in cues if c['cue']=='loop'),None)
        if charge:weapon['weaponclass']['firesound']=q(charge['file']);charge['runtime_connection']='WeaponClass.fireSound charge tone'
        elif kind=='burst':weapon['weaponclass']['firesound']='NULL'
        for level in range(1,settings['ordnanceCount']+1):
            active=level>1;target=ord_
            if active and kind=='charge_beam':
                target=ord_+str(level);child=copy.deepcopy(ord_doc);child['ordnanceclass']['damageflame']=recipe['damage']*(level-1);extra[target]=child
            settings.update({f'shotDelay{level}':0 if level==1 else (.35 if level==2 else 1.),f'salvoCount{level}':(1 if kind=='charge_beam' else 3) if active else 0,f'salvoDelay{level}':.08 if kind=='burst' else 0,f'fireSound{level}':q(fire['file']) if active else 'NULL',f'ordName{level}':q(target) if active else 'NULL',f'wpnReticle{level}':q('gmaggun.'+str(level))})
        weapon['chargegunclass']=settings
        limits.append('Stock BZR charge/release control; burst fires on release' if kind=='burst' else 'Stock BZR charge levels; Gold timings/damage not recovered')
    if kind=='barrage':
        weapon['targetinggunclass'].update(shotdelay=recipe['delay'],firstdelay=.25,salvodelay=.12,salvocount=6,leadersound=q(fire['file']))
        limits.append('BZR TAG leader/lock and six-shot salvo, not Gold target scheduling')
    if kind=='remote':
        ord_doc['bouncebombclass']={'soundBounce':xpl_doc['ExplosionClass']['explSound']}
        limits.append('BZR MDM remote detonation; EMP system disruption / flak interception are not implemented')
    if kind=='time_fuse':
        ord_doc['ordnanceclass'].update(lifespan=.65,classlabel=q('bullet'));ord_doc.pop('tracerclass',None);limits.append('Physical bullet with fixed .65s expiry fuse, not Gold magnetic proximity sensing')
    if kind in ('emp',) or id_=='emp':
        limits.append('EMP-colored damage pulse; no enemy-system disable status effect')
    if kind=='cluster':
        fragment=ord_+'f';child=copy.deepcopy(ord_doc);child['ordnanceclass'].update(shotgeometry='NULL',lifespan=.35,shotspeed=80,ammocost=0,damageballistic=35,xplground='NULL',xplvehicle='NULL',xplbuilding='NULL',xplexpire='NULL');extra[fragment]=child
        xpl_doc['ExplosionClass']['classLabel']=q('quakeblast');xpl_doc['QuakeBlastClass']=dict(quakeCount=6,quakeClass=q(fragment),quakeMagnitude=0)
        limits.append('ISDFC-style quakeblast child ordnance, finite single generation; scatter needs native firing test')
    if kind=='flame':
        ord_doc['ordnanceclass'].update(lifespan=.6,damageballistic=0,damageflame=recipe['damage']);ord_doc['flamepuffclass']=dict(flameRadius=0)
        ord_doc['Core'].update(renderBase=q('draw_twirl'),textureName=q('bgfire.tga'),startRadius=.18,finishRadius=.9,animateTime=.6)
        limits.append('BZR flamepuff stream; borrowed Gold beam audio because no dedicated player flame cue was resolved')
    if kind=='flare':
        weapon['dispenserclass']['objectclass']=q('boflobj');obj={'GameObjectClass':dict(classLabel=q('flare'),baseName=q('flare'),scrapCost=0,scrapValue=0,maxHealth=150,maxAmmo=100,unitName=q('Gold flare'),heatSignature=100.),'MineClass':dict(lifeSpan=10.),'FlareMineClass':dict(payloadName=q(ord_),fireSound=q(fire['file']),triggerDelay=.2,shotDelay=.2)};extra['boflobj']=obj
        ord_doc['ordnanceclass'].update(lifespan=.5,damageballistic=0,damageflame=0,heatsignature=100.);ord_doc['flamepuffclass']=dict(flameRadius=0);ord_doc['Core'].update(renderBase=q('draw_twirl'),textureName=q('bgglow.tga'),startRadius=.5,finishRadius=.2)
        limits.append('BZR Solar Flare decoy object using stock flare base geometry and Gold firing audio')
    return extra,limits


def tier_aliases(texts,docs,out):
    groups={'CANNON':('c','main'),'AUTOCANNON':('a','auto'),'MACHINE_GUN':('g','gatl'),'ROCKETS_UNGUIDED':('r','rock'),'ROCKETS_LASERGUIDED':('l','laser'),'RAILGUN':('s','rail'),'ARTILLERY':('t','arty'),'MISSILESLOCKON_MULTI':('m','multi'),'MISSILESLOCKON_SINGLE':('h','guid')};result=[]
    for key,value in texts.items():
        for group,(code,profile) in groups.items():
            if not re.fullmatch('WEAPON_'+group+r'(?:_[A-P])?',key):continue
            match=re.search(r'\bMK(\d+)\b',value)
            if not match or int(match[1])<2:continue
            if 'SPLIT' in value:profile='split'
            elif 'EMP' in value:profile='cemp' if group=='CANNON' else profile
            elif 'MAG' in value:profile='amag'
            elif 'BURST' in value:profile='brst'
            id_='bt'+code+f'{int(match[1]):02}';pickup='bu'+code+f'{int(match[1]):02}';weapon=copy.deepcopy(docs['bg'+profile]);weapon['weaponclass']['wpnname']=q(value);(out/(id_+'.odf')).write_text('// Native Gold name; BZR base-profile tuning, not recovered tier statistics.\n'+serialize_odf(weapon))
            powerup=copy.deepcopy(docs['ba'+profile]);powerup['weaponpowerupclass']['weaponname']=q(id_);powerup['gameobjectclass']['unitname']=q(value);(out/(pickup+'.odf')).write_text(serialize_odf(powerup));result.append(dict(weapon=id_,pickup=pickup,native_text_key=key,display_name=value,profile='bg'+profile,stats='Same authored BZR profile values; Gold tier statistics unresolved'))
    return result


def cards(base,out,review):
    root=base/'textures_png/chars__actors_pc/graphics/specialfx'
    sources={'bgflash.tga':'pfx/generic/pfx_explosion_flash.tga.dds.png','bgglow.tga':'pfx/glow/pfx_glow_white.tga.dds.png','bgtrail.tga':'pfx/trails/trail_1.tga.dds.png','bgsmoke.tga':'pfx/smoke dust/pfx_smoke_wispy1.tga.dds.png','bgring.tga':'pfx/static/pfx_ringexplosion.dds.png','bgbeam.tga':'pfx/glow/glow_beam.tga.dds.png','bgfire.tga':'pfx/trails/tri_fire.tga.dds.png'}
    result=[];sheet=Image.new('RGB',(len(sources)*200,240),(25,28,32));draw=ImageDraw.Draw(sheet)
    for i,(name,relative) in enumerate(sources.items()):
        source=root/relative
        with Image.open(source) as im:
            rgba=im.convert('RGBA');processing='Native image RGBA retained; PNG decoded to uncompressed TGA; whole static card'
            if name=='bgsmoke.tga':
                # Visually verified 8x8 sheet. Use one clean middle frame,
                # not 64 tiny smoke puffs on every particle billboard.
                w,h=rgba.size
                if w%8 or h%8:raise ValueError('Unexpected smoke atlas dimensions')
                rgba=rgba.crop((2*w//8,h//8,3*w//8,2*h//8));processing='Native RGBA from smoke atlas cell (column 2,row 1), zero-based 8x8 grid; static frame, no recovered native animation timing'
            rgba.save(out/name)
            thumb=rgba.copy();thumb.thumbnail((180,180));sheet.paste(thumb,(i*200+(200-thumb.width)//2,40),thumb)
            draw.text((i*200+8,10),name,fill='white')
            result.append(dict(file=name,source=str(source),source_sha256=digest(source),size=list(rgba.size),processing=processing))
    sheet.save(review/'effect_texture_cards.png');return result


def lua_helper(rows):
    entries='\n'.join('    ['+q(r['id'])+'] = { weapon = '+q(r['weapon'])+', pickup = '+q(r['pickup'])+' },' for r in rows)
    return '''-- Gold weapon test helpers. Explicit mission calls only; no global callbacks.
-- ISDFC uses GiveWeapon(handle, odf, zeroBasedSlot); use an appropriate hardpoint.
local M = {}
M.weapons = {
'''+entries+'''
}
function M.Equip(handle, id, slot)
    assert(M.weapons[id], "Unknown Gold weapon ID")
    assert(type(slot) == "number" and slot == math.floor(slot) and slot >= 0 and slot <= 4, "slot must be 0..4")
    GiveWeapon(handle, M.weapons[id].weapon, slot)
end
function M.Pickup(id, team, position)
    assert(M.weapons[id], "Unknown Gold weapon ID")
    return BuildObject(M.weapons[id].pickup, team, position)
end
return M
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dump-root',type=Path,required=True);parser.add_argument('--game-root',type=Path,required=True);parser.add_argument('--output-root',type=Path)
    parser.add_argument('--blender',type=Path,default=Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'))
    parser.add_argument('--ffmpeg',default=shutil.which('ffmpeg'));parser.add_argument('--reuse-models',action='store_true')
    parser.add_argument('--toolkit-root',type=Path,default=REPO.parent/'BZ98RBlenderToolKit');parser.add_argument('--toolbox-root',type=Path,default=REPO.parent/'Battlezone98_Modding_Toolbox');parser.add_argument('--reference-root',type=Path,default=REPO.parent/'ISDF_Chronicles-geo-winding')
    args=parser.parse_args();base=args.dump_root.resolve();game=args.game_root.resolve();out=(args.output_root or base/'bzr_weapons/goldweapons').resolve();out.mkdir(parents=True,exist_ok=True);review=out.parent/'review';review.mkdir(exist_ok=True)
    if not args.ffmpeg:raise ValueError('FFmpeg required; pass --ffmpeg')
    stock=game/'StockODFFiles';reference=args.reference_root.resolve()
    evidence=[]
    for file in ('10mm.odf','mini2.odf','4hfire.odf','5xhfire.odf','isdfcflash.lua','isdfcshots.lua'):
        path=reference/file
        if not path.exists():raise ValueError('Missing requested ISDFC example: '+str(path))
        evidence.append(dict(path=str(path),sha256=digest(path)))
    if not args.reuse_models:
        command=[str(args.blender),'--background','--python-exit-code','1','--python',str(REPO/'tools/export-gold-projectiles.py'),'--']
        for key,value in [('dump-root',base),('game-root',game),('output-root',out),('toolkit-root',args.toolkit_root),('toolbox-root',args.toolbox_root)]:command+=['--'+key,str(value)]
        with (out.parent/'build_projectiles.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    projectiles=json.loads((out.parent/'projectiles.json').read_text())
    # A cache must match both glTF and binary; default builds always regenerate.
    for row in projectiles:
        source=Path(row['source'])
        if digest(source)!=row['source_sha256'] or digest(source.with_suffix('.bin'))!=row['source_bin_sha256']:raise ValueError('Stale projectile cache')
    Image.new('RGBA',(4,4),(128,128,255,255)).save(out/'bgflatn.png')
    for material_path in out.glob('bp*.material'):
        material_path.write_text(with_stock_import(material_path.read_text().replace('NormalMap flat_n.png','NormalMap bgflatn.png')))
    # Remove only superseded six-hash aliases produced by the early port worker.
    # Native input textures are outside this generated runtime folder.
    for path in out.glob('g??????_?.png'):
        if re.fullmatch(r'g[0-9a-f]{6}_[dens]\.png',path.name) and (out/(path.name[:6]+path.name[7:])).is_file():path.unlink()
    text_source=next((base/'raw/text__menu__menu_asr_en/HTXT_chunk').glob('*.dat'));texts=read_gold_text(text_source.read_bytes())
    effects=json.loads((base/'resources/effects/index.json').read_text())['groups']
    texture_receipt=cards(base,out,review);rows=[];dependencies={};documents={}
    for recipe in recipes():
        id_=recipe['id'];wpn='bg'+id_;ord_='bo'+id_;xpl='bx'+id_;pickup='ba'+id_;colour=recipe['colour']
        weapon_source=stock/(recipe['weapon_template']+'.odf');ord_source=stock/(recipe['ordnance_template']+'.odf');weapon=template(weapon_source);ord_doc=template(ord_source)
        for source in (weapon_source,ord_source):dependencies[str(source)]=digest(source)
        fire=audio(base,out,args.ffmpeg,recipe,False);impact=audio(base,out,args.ffmpeg,recipe,True);cues=audio_cues(base,out,args.ffmpeg,recipe,fire)
        weapon['weaponclass'].update(ordname=q(ord_),wpnname=q('Gold '+recipe['family'].replace('_',' ')),firesound=q(fire['file']))
        section=next((s for s in ('launcherclass','targetinggunclass','dispenserclass','cannonclass') if s in weapon),'cannonclass');weapon.setdefault(section,{})['shotdelay']=recipe['delay']
        if id_ in ('rock','guid','multi','laser','barr'):weapon['weaponclass']['wpncategory']=q('ROCK')
        if id_ in ('rail','flare'):weapon['weaponclass']['wpncategory']=q('SPEC')
        if id_=='laser':weapon['launcherclass']['targetcount']=1
        projectile=next((p for p in projectiles if p['source_model']==recipe['model'] and p['target_length']==recipe['model_length']),None)
        # Keep behavior fields from known stock templates; replace their render graphs.
        ord_doc={s:v for s,v in ord_doc.items() if s in ('ordnanceclass','tracerclass','missileclass','thermalmissileclass','rocketclass','bouncebombclass','flamepuffclass')}
        if 'rocketclass' in ord_doc:ord_doc['rocketclass']={'flameradius':0,'flareradius':0}
        life=12 if id_=='arty' else (8 if id_ in ('guid','multi','laser') else (5 if id_=='rock' else max(.35,300/recipe['speed'])))
        ord_doc['ordnanceclass'].update(shotgeometry=q(projectile['id']+'.geo') if projectile else 'NULL',shotsound='NULL',rendername=q(ord_+'.Render'),xplground=q(xpl),xplvehicle=q(xpl),xplbuilding=q(xpl),xplexpire=q(xpl),ammocost=recipe['ammo'],lifespan=life,shotspeed=recipe['speed'],damageballistic=recipe['damage'],damageconcussion=0,damageflame=0,damageimpact=0)
        if 'tracerclass' in ord_doc:ord_doc['tracerclass']['tracertexture']=q('bgtrail.tga')
        ord_doc.update(render_sections(ord_,colour,large=id_ in ('arty','rock','guid','multi','laser'),rail=id_=='rail'))
        xpl_doc=explosion_sections(xpl,recipe,colour)
        if impact is None:xpl_doc['ExplosionClass']['explSound']='NULL'
        extra,custom_limits=customize(recipe,weapon,ord_doc,xpl_doc,ord_,xpl,fire,cues)
        if id_ in ('beam','chrg','rail'):
            xpl_doc['Core']['textureName']=q('bgglow.tga');xpl_doc['ExplosionClass']['particleCount2']=1
        if id_ in ('emp','cemp'):
            xpl_doc['Core'].update(startRadius=.3,finishRadius=recipe['blast_radius']*1.4,lifeTime=.6,animateTime=.6);xpl_doc['ExplosionClass']['particleCount3']=16
        if id_=='flak':
            xpl_doc['ExplosionClass'].update(particleCount2=10,particleCount3=24);xpl_doc['Smoke'].update(lifeTime=1.6,animateTime=1.6)
        if id_=='fire':xpl_doc['Core']['textureName']=q('bgfire.tga')
        header='// Gold presentation, authored BZR balance. Native Gold gameplay values remain unresolved.\n'
        for name,doc in {wpn:weapon,ord_:ord_doc,xpl:xpl_doc,**extra}.items():
            (out/(name+'.odf')).write_text(header+serialize_odf(doc));documents[name]=doc
        powerup=template(stock/'apmini.odf');powerup['weaponpowerupclass']['weaponname']=q(wpn);powerup['gameobjectclass']['unitname']=q('Gold '+recipe['family'].replace('_',' '));(out/(pickup+'.odf')).write_text(serialize_odf(powerup))
        documents[pickup]=powerup
        notes=[]
        if id_=='rail':notes.append('Fast physical bullet with a 25m tracer; not hitscan, penetration, charge or pilot-instakill snipershell')
        if id_=='multi':notes.append('BZR image-lock targeting; not recovered Gold burst scheduling')
        if id_=='laser':notes.append('Thermal lock-on stand-in; no laser steering; source Missile mesh is a naming candidate, not a decoded binding')
        if id_=='gatl':notes.append('Cropped loop sample repeated per shot; no spin-up/heat controller')
        if id_ in ('main','pist','arty','80'):notes.append('Generic recovered Projectile mesh scaled for the BZR profile; exact native mesh binding unresolved')
        keywords={'main':('Cannon','Tank_Muzzleflash_1'),'pist':('Cannon',),'auto':('Auto_Cannon','AutoCannon','MGcannon'),'gatl':('machine_gun_tracer','MGCannon'),'rail':('Railgun','Sniper_Muzzleflash'),'arty':('Artillery',),'rock':('Rocket','rocket_Launch'),'guid':('Missile_SingleHoming','Missile_Base'),'multi':('Missile','Missile_Base'),'laser':('LaserGuided','Missile_LaserGuided'),'beam':('BeamGun','Laser_muzzleflash'),'chrg':('BeamGun','Laser_muzzleflash'),'emp':('EMP',),'cemp':('EMP',),'fire':('LavaGun',),'flare':('Flare_V01',)}.get(id_,('Cannon',))
        native_fx=[dict(name=g['name'],components=[c['source'] for c in g['components']]) for g in effects if g['namespace']=='chars__actors_pc' and ':' not in g['name'] and any(k.lower() in g['name'].lower() for k in keywords)]
        rows.append(dict(**recipe,weapon=wpn,ordnance=ord_,explosion=xpl,pickup=pickup,projectile=projectile['id'] if projectile else None,fire_audio=fire,impact_audio=impact,audio_cues=cues,additional_odfs=list(extra),native_description=texts.get(recipe['text_key']),native_effect_candidates=native_fx,limitations=notes+custom_limits,balance_source='Authored BZR prototype tuning, not decoded Gold values',effect_source='Native BZR particle graph authored using ISDFC patterns and recovered Gold texture cards; native FXET simulation is not translated'))
    aliases=tier_aliases(texts,documents,out)
    covered={r['text_key'] for r in rows};descriptions={k:v for k,v in texts.items() if k.startswith('WEAPON_DESC_')}
    missing=set(descriptions)-covered
    if missing:raise ValueError('Uncovered native weapon descriptions: '+', '.join(sorted(missing)))
    coverage=dict(source=str(text_source),source_sha256=digest(text_source),decoded_text_records=len(texts),weapon_descriptions=descriptions,all_weapon_descriptions_have_profile=True,profile_count=len(rows),tier_aliases=aliases,equipment_inventory={k:v for k,v in texts.items() if k.startswith('EQUIPMENT_DESC_')},equipment_scope='Only flare has a baseline runtime countermeasure port. Other equipment abilities are cataloged, not claimed as working weapon profiles.',native_text_limit='Native UI entries can include unused content; text presence alone does not certify shipped/unlocked gameplay. Tier aliases preserve names and mechanic profiles, not recovered tier balance.')
    (out.parent/'coverage.json').write_text(json.dumps(coverage,indent=2))
    with (out/'goldweapons_catalog.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['weapon','pickup','family','native_text_key','status']);writer.writerows((r['weapon'],r['pickup'],r['family'],r['text_key'],'authored BZR profile') for r in rows);writer.writerows((r['weapon'],r['pickup'],r['display_name'],r['native_text_key'],'tier name / base BZR profile stats') for r in aliases)
    tank=template(stock/'avtank.odf')
    for i,(hard,weapon_id) in enumerate([('GC1','main'),('GC2','auto'),('GS1','rail'),('GM1','arty')],1):
        tank['gameobjectclass']['weaponhard'+str(i)]=q(hard);tank['gameobjectclass']['weaponname'+str(i)]=q('bg'+weapon_id)
    (out/'avtank.odf').write_text('// Gold weapon loadout; all other stock Grizzly properties retained.\n'+serialize_odf(tank))
    helper_rows=rows+[dict(id=a['weapon'],weapon=a['weapon'],pickup=a['pickup']) for a in aliases]
    shutil.copy2(REPO/'mods/goldweapons/goldweapons.ini',out/'goldweapons.ini');(out/'bgwtest.lua').write_text(lua_helper(helper_rows))
    cue_bank={r['weapon']:dict(fire=r['fire_audio']['file'],impact=r['impact_audio']['file'] if r['impact_audio'] else None,cues=[{k:c[k] for k in ('cue','file','runtime_connection')} for c in r['audio_cues']]) for r in rows}
    (out/'gold_audio_cues.json').write_text(json.dumps(cue_bank,indent=2))
    (out/'README.txt').write_text(f'Gold Weapons - BZR Prototype\n\n{len(rows)} base/behavior profiles and {len(aliases)} native tier-name aliases. All 19 native weapon descriptions have a BZR profile; native text may include unused content. Numeric Gold tier tuning is unresolved; aliases use authored BZR base-profile statistics.\n\nEnable alongside updated Gold Grizzly. Overrides avtank.odf with main cannon/autocannon/railgun/artillery, preserving other stock vehicle fields. Stock weapon ODFs are not replaced.\n\nMission helper: local gold = require("bgwtest"); gold.Equip(GetPlayerHandle(), "main", 0)\nAdditional IDs: beam chrg flak emp fire split cemp amag brst barr flare. Tier aliases can be passed by weapon ID (e.g. bta05). See goldweapons_catalog.csv for all IDs and pickups. Choose a compatible hardpoint; Grizzly has no ROCK socket. Gold Grizzly fixed external gun models do not dynamically swap.\n\nNative BZR charge/release, remote MDM detonation, TAG salvo, flamepuff and flare decoy classes approximate Gold mechanics. EMP disable, flak interception, true laser guidance, exact Gold heat/reload/state behavior remain unimplemented. See gold_audio_cues.json for active sounds and retained lifecycle cues awaiting a controller.\n\nEffects use native BZR graphs and recovered Gold image cards. Source FXET simulation and detailed source gun animations remain archival. No game is launched by this build. Native firing, effects, audio and lifecycle need runtime testing.\n')
    receipt=dict(weapons=rows,tier_aliases=aliases,coverage='coverage.json',projectiles='projectiles.json',textures=texture_receipt,reference_examples=evidence,stock_templates=dependencies,scope='Offline native BZR package; game loading, particles, listening, collision and gameplay need qualification',limitations=['Overrides avtank.odf loadout only; no stock weapon overrides, dynamic external gun swaps or detail animation conversion','Muzzle sprite follows the ISDFC one-shot ordnance emitter pattern; native timing/placement needs runtime qualification','Source filename matching is candidate provenance, not decoded Gold event/mesh bindings'])
    (out.parent/'build.json').write_text(json.dumps(receipt,indent=2));hashes={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()};(out.parent/'runtime_sha256.json').write_text(json.dumps(hashes,indent=2))
    with zipfile.ZipFile(out.parent/'goldweapons.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:archive.write(out/name,'goldweapons/'+name)
    print(json.dumps(dict(weapons=len(rows),projectile_models=len(projectiles),runtime_files=len(hashes),zip=str(out.parent/'goldweapons.zip')),indent=2))


if __name__=='__main__':main()
