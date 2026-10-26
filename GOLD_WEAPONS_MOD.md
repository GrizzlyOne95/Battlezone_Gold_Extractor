# Gold weapons for BZR

`Gold Weapons - BZR Prototype` is a standalone `mapType = "mod"` package. The expanded build contains **22 behavior profiles, 111 native tier-name aliases, 133 pickups, eight projectile model variants, seven recovered effects cards and 100 WAV files**. Its profiles cover all **19 `WEAPON_DESC_*` entries** in Gold's decoded English menu table, plus pistol, classic cannon and flare equipment. Native text can contain unused content; this is evidence of archived names/descriptions, not proof that every entry is available in the shipped game. Enable it with the updated **Gold Grizzly - Tank Model Prototype** to combine tank visuals and weapon behavior. No Workshop upload is performed.

The mod overrides `avtank.odf` with a fixed four-weapon loadout. All other stock Grizzly properties are retained. Stock weapon ODFs are not replaced. Disable Gold Weapons to restore the lower-priority loadout. Enabling it affects Grizzlies using this ODF, including AI. Keep it above other `avtank.odf` replacements; use a separate session for integration with larger mods.

| Gold family | Weapon / pickup | BZR behavior | Default Grizzly socket |
| --- | --- | --- | --- |
| Main cannon | `bgmain` / `bamain` | Direct cannon, recovered generic projectile | GC1 |
| Pistol | `bgpist` / `bapist` | Lighter direct cannon | — |
| Autocannon | `bgauto` / `baauto` | Machinegun, tracer | GC2 |
| Gatling | `bggatl` / `bagatl` | Faster machinegun, tracer | — |
| Railgun | `bgrail` / `barail` | Fast physical bullet and long tracer, SPEC category | GS1 |
| Artillery | `bgarty` / `baarty` | Mortar/grenade, blast damage | GM1 |
| Unguided rockets | `bgrock` / `barock` | Direct cannon firing rocket model | — |
| Guided missiles | `bgguid` / `baguid` | Thermal launcher/missile | — |
| Multi missiles | `bgmulti` / `bamulti` | Image launcher/image missile | — |
| Laser missiles | `bglaser` / `balaser` | Thermal lock-on stand-in | — |
| Classic cannon | `bg80` / `ba80` | Direct cannon variant | — |
| Laser | `bgbeam` / `babeam` | Native beamgun / beam, 180 m authored range | — |
| Charged laser | `bgchrg` / `bachrg` | Native chargegun with two firing levels and Gold charge/release sounds | — |
| Flak cannon | `bgflak` / `baflak` | MDM remote airburst cloud; no projectile interception | — |
| EMP cannon | `bgemp` / `baemp` | MDM remote pulse bomb; no system-disable debuff | — |
| Flame thrower | `bgfire` / `bafire` | Short flamepuff stream; Gold beam audio fallback | — |
| Cluster cannon | `bgsplit` / `basplit` | Single-generation quakeblast with six damaging fragment projectiles | — |
| EMP cannon round | `bgcemp` / `bacemp` | Cannon pulse damage; no system-disable debuff | — |
| MAG autogun | `bgamag` / `baamag` | Physical bullet with fixed 0.65 s expiry fuse | — |
| Burst autogun | `bgbrst` / `babrst` | Three-shot chargegun release burst | — |
| Missile barrage | `bgbarr` / `babarr` | Native TAG leader/targeting and six-missile salvo | — |
| Flare equipment | `bgflare` / `baflare` | Solar Flare heat decoy; stock object geometry, Gold sound/card | — |

The original four default Grizzly weapons are retained. Expanded cannon/beam/flame/burst variants use CANN sockets; railgun and flares use SPEC; artillery and remote EMP/flak bombs use MORT; launchers and barrage use ROCK. The Grizzly has no ROCK hardpoint.

### Native names and tiers

`coverage.json` records all 2,714 English menu entries, verified by hash/key pairing and exact table bounds. The HTXT layout is a counted hash/UTF-16 value table, a padded table name, and counted null-terminated key bytes. Each key must hash to its corresponding value record. This recovers actual descriptions such as the autogun's magnetic time-fused rounds, cannon cluster shells, remote EMP bombs, and flare countermeasures without guessing their meaning from mesh filenames.

The 111 tier aliases preserve native MK names and use the matching behavior profile: for example `bta05` is AutoGun MK5 MAG using `bgamag`, and `btc06` is Cannon MK6 SPLIT using `bgsplit`. Their pickups use the same suffix with `bu` instead of `bt`. **Aliases do not claim recovered tier statistics**; they use the authored BZR profile's damage/ammo/cadence unchanged. `goldweapons_catalog.csv` enumerates every base and tier weapon/pickup ID. The Lua helper accepts both short profile IDs and tier weapon IDs.

The other `EQUIPMENT_DESC_*` abilities are cataloged separately in `coverage.json`. Vampire, virus, shield boost, cloaking and similar status/controller abilities are not silently represented as working ordinary guns. Only flare equipment has a baseline countermeasure implementation in this build.

The Gold Grizzly VDF/skeleton now maps **GS1 to the railgun muzzle and GM1 to artillery**, matching SPEC/MORT categories. Rebuild/install the tank package as well as the weapons when upgrading from the earlier opposite socket mapping. Its fixed visible guns do not swap with pickup changes. Pickups use BZR's stock cannon pickup geometry; they are functional weapon pickups, not recovered Gold pickup models.

## Using other weapons

A mission can load the supplied short, unique Lua module without installing any callbacks:

```lua
local gold = require("bgwtest")
gold.Equip(GetPlayerHandle(), "main", 0) -- zero-based weapon slot
gold.Pickup("guid", 1, GetPosition(GetPlayerHandle()))
```

Profile IDs are `main`, `pist`, `auto`, `gatl`, `rail`, `arty`, `rock`, `guid`, `multi`, `laser`, `80`, `beam`, `chrg`, `flak`, `emp`, `fire`, `split`, `cemp`, `amag`, `brst`, `barr`, `flare`. A tier can be equipped as `gold.Equip(handle, "bta05", slot)`. Use a compatible hardpoint. These calls belong in an authored mission/test harness, not an automatically loaded global script. No EXU dependency is required for the baseline particles.

## Effects and audio

Every ordnance has an explicit native `draw_multi` graph: tracer/core, flight trail and a one-shot muzzle emitter. Its ground/vehicle/building/expiry events reference its own explosion ODF. Each explosion spawns a core, smoke and sparks and plays its Gold impact WAV. Impact-only profiles have zero blast damage; explosive profiles put their blast in the explosion instead of duplicating ordnance concussion damage.

The patterns reference the local **ISDF Chronicles** examples `10mm.odf`, `mini2.odf`, `4hfire.odf`, `5xhfire.odf`, `isdfcflash.lua` and `isdfcshots.lua`. Exact input paths and SHA-256 values are in `bzr_weapons/build.json`. This package authors its own native particle graphs; it does not copy ISDFC's EXU dispatcher or depend on its global callbacks. ISDFC's separate EXU `draw_geom` muzzle system was inspected but is not used: Redux's baseline does not gain a functioning 3D muzzle flash merely from `flashName` fields.

Every projectile `.material` explicitly imports `BZBase.material`, including cached model reuse. The first native tank test exposed missing imports in the shared material generator; all eight projectile scripts were repaired along with the tank. The validators now check external material/abstract-pass inheritance visibility before installation.

Seven native Gold RGBA images supply flash, glow, trail, smoke, impact ring, beam and flame cards. The smoke source is an **8×8 animation atlas**; the port selects the single frame at zero-based column 2 / row 1. The full sheet is retained in the original dump. This is a static smoke approximation, not recovered native frame timing. Additive graphs use `srcalpha one modulatealpha`, as used by stock BZR effects, because these cards can have white RGB outside their alpha silhouette. Unmasked additive blending would produce bright squares. Beam, flame, EMP and flak profiles have different render/impact recipes rather than sharing one generic cannon presentation. Candidate native FXET effect groups and component paths are retained per weapon in the build receipt.

The audio bank now includes **43 primary firing/impact files and 57 lifecycle cue files**. Flare has no invented impact sound. Charge, loop, stop, tail, reload, stow, unstow and dry-fire cues are preserved where found; `gold_audio_cues.json` says which are connected and which await an event controller. The charged laser connects its native loop as the charge tone and its stop/release cue to charge-level fire sounds. Most reload/stow/tail cues are retained assets, not automatically synchronized sounds.

Primary firing sounds are cropped/faded for authored cadence and resampled to mono PCM16 at 22,050 Hz. Impact and lifecycle WAVs retain original duration. The gatling uses a loop segment per accepted shot. Multi/laser/barrage profiles reuse a guided launch cue; classic and flak use shared cannon audio; flame uses beam audio because no dedicated player flame cue was resolved. Railgun impact uses the charged-laser category candidate. The flare cue comes from `chars__actors_pc`, which the earlier shared-bank-only search missed. Filename/category matches establish candidate provenance, not verified Gold event routing. Validators check every file's format, nonzero samples, peak/RMS and source hash; cadence, loudness, overlap and positional audio still need in-game listening.

## Models and fidelity limits

The worker exports five recovered mesh sources as eight explicitly sized model variants, retaining each complete projectile shape/fins and its material assignments. It measures **evaluated** bind geometry before sizing, filters meshes against the actual glTF mesh list, and excludes unrelated diagnostic Icospheres. Outputs include real `.geo` files plus Ogre `.mesh`/`.skeleton` and material scripts, serialized through the shared Blender toolkit. Legacy GEO texture names fit twelve characters plus the terminator; full source material IDs remain in Ogre material names. A package-local flat normal fallback avoids an implicit stock texture dependency.

Source node transforms, source `.gltf`/`.bin` hashes, axis conversion, size, repairs and material constants are recorded in `projectiles.json`. UVs use Blender `(U,V)` → Ogre `(U,1−V)` and the review importer reverses this. Projectile details are frozen in their source bind pose; native rigs/clips remain in the extraction libraries.

All damage, range, speed, cadence, blast radius, scale and effect colors are **authored BZR prototype values** in `tools/gold_weapons.py`. Native Gold instance tuning and inherited blueprint layouts remain unresolved. Railgun is a fast projectile with no penetration or pilot-instakill behavior. Gatling lacks spin-up/heat. Multi missiles use BZR image lock. The laser missile profile has no laser steering. EMP profiles deal pulse damage but do not disable systems; flak does not intercept enemy ordnance; the MAG fuse is a fixed expiry rather than magnetic proximity. Burst firing uses BZR charge/release controls. Generic cannon/artillery projectiles and the laser profile's `Missile` model remain presentation candidates rather than decoded native bindings. Native FXET simulation and DXBC are research sources rather than executed by BZR.

### Ordnance orientation against existing assets

`audit-gold-ordnance.py` reads the emitted Ogre mesh vertices and compares four installed BZR ordnance references (`gatst00`, `ghorn00`, `gshad00`, `ghartl00`) plus ISDFC's `iomshd00`. Gold's eight variants use +Z flight, +Y up and centered origins, matching the stock references. The ISDFC model has an end origin; the report preserves that difference instead of forcing Gold to copy it. The annotated side projections show each origin and forward arrow.

Independent native landmarks check sign, not just longest-axis alignment: body-minus-tail for the unguided rocket; front-minus-rear wings for guided/multi missiles; rear thrust locations for other projectiles. A socket's local basis is not automatically a flight axis: the native `Missile` thrust socket has a sideways local Z while its rear location and mesh clearly establish longitudinal flight. The audit therefore uses its rear anchor, avoiding an incorrect 90-degree rotation. All eight exported variants pass; none required a flip. Their largest principal-axis deviation is about 0.368 degrees, compared with about 0.805 degrees for the sampled stock references. This measures vertex distribution, not aim error.

`orientation_validation.json` stores model/source hashes, bounds, centers and landmark checks; `review/ordnance_orientation.png` is the comparison. Validation also compares all GEO and Ogre vertex positions, so the legacy geometry cannot silently disagree with the visible mesh. This is offline qualification, not a native flight/collision test.

## Build, validate, review and install

Run from the extractor repository with Python, Pillow/NumPy, FFmpeg and Blender 5.2 available:

```powershell
$dumpRoot = 'work/asset_dump_2026-10-24'
$gameRoot = 'C:/Program Files (x86)/GOG Galaxy/Games/Battlezone 98 Redux'
python tools/build-gold-weapons-mod.py --dump-root $dumpRoot --game-root $gameRoot
python tools/audit-gold-ordnance.py --mod-root "$dumpRoot/bzr_weapons/goldweapons" --game-root $gameRoot
python tools/validate-gold-weapons-mod.py --mod-root "$dumpRoot/bzr_weapons/goldweapons" --game-root $gameRoot
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --python-exit-code 1 --python tools/review-gold-weapons-mod.py -- --mod-root "$dumpRoot/bzr_weapons/goldweapons"
python tools/install-gold-grizzly-mod.py --mod-root "$dumpRoot/bzr_weapons/goldweapons" --game-root $gameRoot --mod-name goldweapons
python -m unittest discover -s tests -v
```

Pass `--ffmpeg PATH`, `--blender PATH`, `--reference-root PATH`, `--toolkit-root PATH` or `--toolbox-root PATH` to the builder when needed. The orientation figure also requires Matplotlib (`pip install matplotlib`). `--reuse-models` is for ODF/audio/effect iteration only; it checks source glTF/bin hashes but does not validate an unchanged exporter recipe. Regenerate models and rerun the orientation audit after changing exporter code, pose, materials or scale.

Outputs under `work/asset_dump_2026-10-24/bzr_weapons/`:

- `goldweapons/` and `goldweapons.zip`: flat standalone runtime package.
- `build.json`, `projectiles.json`, `runtime_sha256.json`, `validation.json`, `installation.json`: provenance and checks outside the runtime folder.
- `review/projectile_gallery.png`, `gold_projectiles_review.blend`, `projectile_review.json`: reloaded exported mesh review.
- `review/effect_texture_cards.png`: converted source-card inspection, not a simulation of BZR particles.

The validator checks weapon/pickup/ordnance/explosion links, render section closure/counts/cycles, alpha-aware blend modes, every local effect texture, all audio links/formats, Grizzly loadout/category alignment and unchanged non-loadout fields, model/skeleton/GEO/material links, stock material imports, finite geometry/UVs, unit normals, exact target lengths and GEO/mesh triangle counts, file hashes and ZIP bytes. It does not run BZR or prove native renderer/gameplay behavior. Installation copies only `addon/goldweapons` and preserves current mod selection. Native qualification still needs firing each profile, aim/muzzle alignment, collision, lock-on, effects, listening, pause/save/load and pickup tests.
