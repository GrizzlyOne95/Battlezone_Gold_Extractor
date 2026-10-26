"""BZR weapon recipes and strict ODF/reference helpers (no Blender dependency)."""
import re
import struct


def read_gold_text(data):
    """HTXT: hash/UTF-16 values, padded table name, counted ASCII key bytes."""
    if len(data)<16:raise ValueError('Truncated HTXT')
    count,language,value_bytes,flags=struct.unpack_from('<4I',data)
    names_offset=16+count*8+value_bytes
    if count>100000 or not 16<=names_offset<=len(data):raise ValueError('Invalid HTXT bounds')
    cursor=16;values=[]
    for _ in range(count):
        if cursor+8>names_offset:raise ValueError('Truncated HTXT record')
        hash_,length=struct.unpack_from('<2I',data,cursor);cursor+=8;end=cursor+length*2
        if not length or end>names_offset or data[end-2:end]!=b'\0\0':raise ValueError(f'Invalid HTXT string at record {_}, offset {cursor}, length {length}, end {end}, names {names_offset}')
        values.append((hash_,data[cursor:end-2].decode('utf-16le')));cursor=end
    if cursor!=names_offset:raise ValueError('HTXT value table not consumed')
    label_end=data.index(b'\0',names_offset)
    size_offset=names_offset+((label_end-names_offset+4)//4)*4
    if size_offset+4>len(data):raise ValueError('Truncated HTXT key table')
    key_bytes=struct.unpack_from('<I',data,size_offset)[0];names_offset=size_offset+4
    if names_offset+key_bytes!=len(data):raise ValueError('HTXT key byte length mismatch')
    names=data[names_offset:].split(b'\0')
    if names[-1]!=b'' or len(names)!=count+1:raise ValueError('HTXT key count mismatch')
    result={}
    for name,(hash_,value) in zip(names,values):
        key=name.decode('cp1252');computed=0
        for char in name.lower():computed=(31*computed+char)&0xffffffff
        if computed!=hash_:
            signed=0
            for char in name.lower():signed=(31*signed+(char if char<128 else char-256))&0xffffffff
            if signed!=hash_:raise ValueError('HTXT key/hash order mismatch: '+key)
        if key in result:raise ValueError('Duplicate HTXT key')
        result[key]=value
    return result


# All tuning below is an authored BZR starting point, never recovered Gold balance.
# Family, short ID, weapon/ordnance templates, mesh, target mesh length, cadence,
# speed, ammo, ballistic damage, blast damage, blast radius, firing/impact tokens.
RECIPES = [
    ('Main_Gun','main','gatstab','rocket','projectile',.65,.55,170,12,180,35,2.0,'/cannon/','/cannon/'),
    ('Pistol_Weapon','pist','gatstab','rocket','projectile',.30,.30,200,4,70,0,0,'/pistol/','/cannon/'),
    ('Autocannon','auto','gminigun','bullet1',None,0,.20,220,2,45,0,0,'/auto_cannon_02/','/machine gun/'),
    ('Gatling_Gun','gatl','gminigun','bullet1',None,0,.08,240,1,22,0,0,'/machine_gun/','/machine gun/'),
    ('Railgun','rail','gatstab','rocket',None,0,1.6,1000,35,600,0,0,'/railgun/','/chargedlaser/'),
    ('Artillery_weapon','arty','gmortar','grenade','projectile',.8,2.0,55,50,50,350,8.0,'/artillery/','/artillery/'),
    ('Unguided_rocket_launcher','rock','gatstab','rocket','unguided_rocket',1.2,.9,85,20,130,90,4.0,'/unguided_rocket/','/rocket/'),
    ('Guided_Missile_Launcher','guid','ghornet','heatmsl','guided_missile',1.3,1.3,55,30,180,100,4.0,'/guided_missile/','/missile/'),
    ('Multi_missile_launcher','multi','gshadow','imagemsl','multi_missile',.8,1.5,65,15,80,40,2.5,'/guided_missile/','/missile/'),
    ('Laser_missile_launcher','laser','ghornet','heatmsl','missile',1.2,1.5,80,35,240,100,4.5,'/guided_missile/','/missile/'),
    ('PlayerTank_Weapon_Cannon_80s','80','gatstab','rocket','projectile',.6,.6,160,12,180,35,2.0,'/cannon/','/cannon/'),
    ('Laser_Beam','beam','gflashgn','flash',None,0,.06,1e6,5,22,0,0,'/beam_gun/','/laser/'),
    ('Charged_Laser','chrg','gmaggun','flash',None,0,.7,1e6,35,240,0,0,'/chargedlaser/','/chargedlaser/'),
    ('Flak_Cannon','flak','gmdmgun','bouncebm','projectile',.65,.9,75,20,20,60,5.,'/cannon/','/cannon/'),
    ('EMP_Cannon','emp','gmdmgun','bouncebm','projectile',.65,1.3,70,40,0,120,7.,'/emp/','/emp/'),
    ('Flame_Thrower','fire','gminigun','flame',None,0,.08,35,1,14,0,0,'/beam_gun/','/laser/'),
    ('Cannon_Split','split','gatstab','rocket','projectile',.65,.85,160,20,80,60,3.,'/cannon/','/cannon/'),
    ('Cannon_EMP','cemp','gatstab','rocket','projectile',.65,.8,150,25,50,65,4.,'/cannon/','/emp/'),
    ('Autocannon_MAG','amag','gminigun','bullet1',None,0,.2,180,3,40,15,1.5,'/auto_cannon_02/','/machine gun/'),
    ('Autocannon_Burst','brst','gmaggun','bullet1',None,0,.65,220,2,38,0,0,'/auto_cannon_02/','/machine gun/'),
    ('Missile_Barrage','barr','gtaggun','swarmer','multi_missile',.8,1.6,60,12,70,35,2.5,'/guided_missile/','/missile/'),
    ('Flare_Dispenser','flare','gflare','flame',None,0,2.,12,3,0,0,0,'/flare/',None),
]

TEXT_KEYS={'main':'WEAPON_DESC_CANNON','auto':'WEAPON_DESC_AUTOCANNON','gatl':'WEAPON_DESC_MACHINE_GUN','rock':'WEAPON_DESC_ROCKETS_UNGUIDED','guid':'WEAPON_DESC_MISSILESLOCKON_SINGLE','multi':'WEAPON_DESC_MISSILESLOCKON_MULTI','laser':'WEAPON_DESC_ROCKETS_LASERGUIDED','rail':'WEAPON_DESC_RAILGUN','arty':'WEAPON_DESC_ARTILLERY','beam':'WEAPON_DESC_LASER','chrg':'WEAPON_DESC_CHARGED_LASER','flak':'WEAPON_DESC_FLAKCANNON','emp':'WEAPON_DESC_EMPCANNON','fire':'WEAPON_DESC_FLAMETHROWER','split':'WEAPON_DESC_CANNON_SPLIT','cemp':'WEAPON_DESC_CANNON_EMP','amag':'WEAPON_DESC_AUTOCANNON_MAG','brst':'WEAPON_DESC_AUTOCANNON_BURST','barr':'WEAPON_DESC_MISSILES','flare':'EQUIPMENT_DESC_FLARE','pist':'CODEX_WEAPON_PISTOL_DESC'}
KINDS={'beam':'beam','chrg':'charge_beam','brst':'burst','flak':'remote','emp':'remote','fire':'flame','flare':'flare','split':'cluster','cemp':'emp','amag':'time_fuse','barr':'barrage'}
COLOURS={'beam':(100,255,150),'chrg':(170,110,255),'emp':(120,150,255),'cemp':(120,150,255),'fire':(255,120,40),'flare':(255,180,100),'flak':(255,200,110)}


def recipes():
    fields=('family','id','weapon_template','ordnance_template','model','model_length','delay','speed','ammo','damage','blast_damage','blast_radius','fire_token','impact_token')
    rows=[dict(zip(fields,r)) for r in RECIPES]
    for row in rows:
        id_=row['id'];row.update(kind=KINDS.get(id_,'projectile'),text_key=TEXT_KEYS.get(id_),colour=COLOURS.get(id_,(255,120,80) if id_ in ('80','rock') else (100,210,255)))
    return rows


def parse_odf(text):
    """Read BZR's section/key subset without ConfigParser interpolation assumptions."""
    result={};section=None
    for line in text.splitlines():
        line=line.split('//',1)[0].strip()
        if not line:continue
        if line.startswith('[') and line.endswith(']'):
            section=line[1:-1].strip().lower()
            if section in result:raise ValueError('Duplicate ODF section: '+section)
            result[section]={}
        elif '=' in line and section:
            key,value=line.split('=',1);key=key.strip().lower()
            if key in result[section]:raise ValueError('Duplicate ODF key: '+key)
            result[section][key]=value.strip().strip('"')
    return result


def serialize_odf(sections):
    return '\n\n'.join('['+name+']\n'+'\n'.join(k+' = '+str(v) for k,v in values.items()) for name,values in sections.items())+'\n'


def q(value):return '"'+str(value)+'"'


def verify_render_graph(documents,roots):
    """Reject missing sections, draw_multi count mismatches and render cycles."""
    visited=set();active=set()
    def walk(ref):
        ref=ref.lower()
        if ref in active:raise ValueError('Render cycle: '+ref)
        if ref in visited:return
        if '.' not in ref:raise ValueError('Expected file.section: '+ref)
        file,section=ref.split('.',1)
        if file not in documents or section not in documents[file]:raise ValueError('Missing render section: '+ref)
        values=documents[file][section];base=values.get('renderbase','')
        if base not in ('draw_multi','draw_emit','draw_twirl','draw_twirl_trail','draw_trail','draw_tracer','draw_light','draw_bolt'):raise ValueError('Unsupported render: '+base)
        active.add(ref)
        if base=='draw_multi':
            count=int(values['rendercount']);refs={k for k in values if re.fullmatch(r'rendername\d+',k)}
            if refs!={'rendername'+str(i) for i in range(1,count+1)}:raise ValueError('draw_multi count mismatch: '+ref)
            for i in range(1,count+1):walk(values['rendername'+str(i)])
        if base=='draw_emit':walk(values['emitname'])
        active.remove(ref);visited.add(ref)
    for root in roots:walk(root)
    return visited


def render_sections(stem,colour,large=False,rail=False):
    """ISDFC-style native particle graphs, with recovered Gold texture cards."""
    rgb=' '.join(map(str,colour));radius=.30 if large else .09
    return {
        'Render':dict(renderBase=q('draw_multi'),renderCount=3,renderName1=q(stem+'.Core'),renderName2=q(stem+'.Trail'),renderName3=q(stem+'.Flash')),
        'Core':dict(renderBase=q('draw_tracer'),textureBlend=q('srcalpha one modulatealpha'),textureName=q('bgtrail.tga'),startColor=q(rgb+' 255'),finishColor=q(rgb+' 0'),startRadius=radius,finishRadius=radius,tracerLength=25 if rail else (1.0 if large else 4.0)),
        'Trail':dict(renderBase=q('draw_twirl_trail' if large else 'draw_trail'),textureName=q('bgsmoke.tga' if large else 'bgtrail.tga'),textureBlend=q('srcalpha invsrcalpha modulatealpha' if large else 'srcalpha one modulatealpha'),startColor=q('160 180 200 100' if large else rgb+' 150'),finishColor=q(rgb+' 0'),startRadius=.18 if large else .06,finishRadius=.65 if large else .025,animateTime=.5,emitDelay=.025,emitLife=.5,emitVelocity=q('0.0 0.0 0.0'),emitVariance=q('0.1 0.1 0.1'),segmentTime=.15,textureRate=.25),
        # The emitter spawns once at shot creation, then never again during flight.
        'Flash':dict(renderBase=q('draw_emit'),emitName=q(stem+'.Muzzle'),emitDelay=1e6,emitVelocity=q('0.0 0.0 0.0'),emitVariance=q('0.0 0.0 0.0')),
        'Muzzle':dict(simulateBase=q('sim_smoke'),lifeTime=.08,renderBase=q('draw_twirl'),textureName=q('bgflash.tga'),textureBlend=q('srcalpha one modulatealpha'),startColor=q(rgb+' 220'),finishColor=q(rgb+' 0'),startRadius=.65 if large else .30,finishRadius=.12,animateTime=.08,rotationRate=0,BottomInteractsWithTerrain='false'),
    }


def explosion_sections(stem,recipe,colour):
    rgb=' '.join(map(str,colour));large=recipe['blast_radius']>0;r=max(.5,recipe['blast_radius']*.6)
    sections={'ExplosionClass':dict(classLabel=q('explosion'),explSound=q('bg'+recipe['id']+'i.wav'),damageRadius=recipe['blast_radius'],damageBallistic=0,damageConcussion=recipe['blast_damage'],damageFlame=0,damageImpact=0,particleTypes=3,
        particleClass1=q(stem+'.Core'),particleCount1=1,particleVeloc1=q('0.0 0.0 0.0'),
        particleClass2=q(stem+'.Smoke'),particleCount2=3 if large else 1,particleVeloc2=q('1.0 2.0 1.0'),
        particleClass3=q(stem+'.Spark'),particleCount3=8 if large else 4,particleVeloc3=q('4.0 5.0 4.0'))}
    for name,tex,size,life,blend in [('Core','bgring.tga',r,.35,'srcalpha one modulatealpha'),('Smoke','bgsmoke.tga',r*.8,.8,'srcalpha invsrcalpha modulatealpha'),('Spark','bgglow.tga',.08,.3,'srcalpha one modulatealpha')]:
        sections[name]=dict(simulateBase=q('sim_smoke'),lifeTime=life,renderBase=q('draw_twirl'),textureName=q(tex),textureBlend=q(blend),startColor=q(rgb+' 220'),finishColor=q(rgb+' 0'),startRadius=size*.3,finishRadius=size,animateTime=life,rotationRate=2,BottomInteractsWithTerrain='false')
    return sections
