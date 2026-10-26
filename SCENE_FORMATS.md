# Structured scene recovery

Validated against the local Steam installation on October 24, 2026. This is a reverse-engineered format description, not an official engine specification. The older counts and heuristic offsets in `ASURA_FORMATS.md` are historical; these readers supersede its geometry assumptions.

## Render resources

Named, extensionless RSCF render resources carry the actual render triangles, material slots, UVs, normal data, skin weights and joint indices.

- `0x00`: four u32 values: submesh count, vertex count, index count, triangle count.
- `0x10`: u32 bone-palette count.
- Submesh descriptors start at `0x14`, with 24 bytes each. Material hash is descriptor +0; index count is descriptor +8; palette index is descriptor +20.
- Six f32 values at `20 + submesh_count*24`: quantization scale XYZ, then offset XYZ.
- Vertex buffer starts at **`44 + submesh_count*24`**, with 40 bytes per vertex. The old exporter incorrectly started 20 bytes earlier and treated the difference as a trailing gap.
- Vertex +0: XYZ as three u16 values. Position is `u16 / 65536 * scale + offset`.
- Vertex +8: packed normal, three unsigned 10-bit fields mapped to [-1,1].
- Vertex +12/+16: native packed tangent/binormal, retained in the original render resource. Blender currently computes tangents from the recovered UVs.
- Vertex +24 and +28: two pairs of IEEE float16 UV coordinates.
- Vertex +32: four normalized byte weights; +36: four byte joint indices.
- The u16 triangle-list index buffer follows the vertices immediately.
- Count-prefixed bone palettes follow the index buffer. Joint bytes already refer to global skeleton nodes; these palettes must not remap them.

The reader checks exact resource size, submesh totals, triangle counts, weight sums, valid indices, palette bounds and joint references. Nonfinite UV components are reported with their exact source half-float bits; interchange-only components become zero because glTF cannot represent infinity. One MiningSpider source has 149 such components. For the player tank, per-joint render bounds agree with independent HSKN CPU geometry bounds to within quantization precision.

## HSKN hierarchy

- Two u32 values: flags and node count.
- One padded owner name.
- `node_count` u32 parent indices. Root refers to itself; later parents precede children.
- `node_count` transforms, each seven f32 values: local translation XYZ and quaternion XYZW.
- One geometry-present byte.
- For each node: padded node name, node flag byte, u32 geometry byte length, and that many bytes of CPU geometry.
- Node geometry version 14 has a 36-byte prefix including its size, six bounds floats, one radius float, then XYZ float vertices. The old decoder interpreted the radius as the first vertex component.
- CPU faces are **four u16 slots per face**, with `0xFFFF` in the fourth slot for a triangle. These are not a continuous triangle strip.
- Ancillary collision and resource tables after the node data remain in the raw source. Render geometry is taken from the named render resource, not reconstructed from collision faces.

Both observed node flag values (0 and 1) are supported and retained in extras. Other values fail explicitly. Node-only marker resources are exported as hierarchies without inventing visible geometry.

## Materials and sockets

The native name hash is a lowercase polynomial hash: start at zero, then `hash = hash*31 + character`, modulo 2^32. Texture IDs hash `/path/without/graphics/or/extension` using forward slashes. This resolves exact texture references rather than ranking similar filenames.

MARE records retain all 89 u32 words. Common texture fields are base color +0x0C, normal +0x10, specular +0x14, opacity +0x18, emissive +0x1C, reflection +0x44. These fields are shader-dependent. Core glTF carries base color, normal, emissive and dedicated alpha maps; native shader parameters remain in extras and raw MARE data. PBR roughness/metalness behavior is an approximation, not a verified reproduction of the game shader.

Core glTF normal maps use a derived 8-bit OpenGL preview: reconstruct positive Z from native packed RG and invert green for Blender's tangent convention. The original packed texture reference remains in material extras and the texture dump. The Blender material builder decodes the original packed texture directly in its node group, avoiding the derived preview's additional quantization.

HMPT contains an owner name and attachment records: translation XYZ, a 3x3 basis, parent node index, padded socket name. Native sockets, including muzzle and screen markers, are exported as separate nodes. The cockpit assembly's component-to-socket aliases are listed explicitly; their pairing has not been compared to the running game. All source CONA/ENTI state and entity data remain in the raw dump.

## HCAN animation

See [ANIMATIONS.md](ANIMATIONS.md) for native clip inventories, Blender playback commands and the distinction between preserved clips and motion-tested clips.

- Header: named track count, duration f32, a motion parameter f32, quaternion-key count, translation-key count, flags.
- Padded clip name.
- Each named track is `<HHII>`: rotation count, translation count, rotation start index, translation start index.
- A separate root-motion track is present when named-track totals do not account for the key arrays. The reader checks that adding it accounts for all keys.
- Rotation keys: four u16 values. XYZ decode as `(value-32767)/32767`; W as `value/65535`, followed by normalization.
- Quaternion times: u16 normalized to clip duration.
- Translation keys: three f32 values each, followed by normalized u16 times.
- The supported footer contains native node-name hashes before the event table. Unsupported footer/event layouts fail explicitly. All discovered clips in this installation use the supported footer.

Clips bind by native node hashes, preserving local hierarchy and separate root motion. Every source key time remains in glTF. Additional 60 Hz spherical rotation samples prevent Blender's component-wise quaternion F-curves from drifting between sparse keys. The assumed linear/spherical interpolation has not been checked against the game runtime.

When an attached resource has both a static calibration matrix and root-motion animation, a separate native-root-motion child carries the animated TRS. The static wrapper retains calibration. This avoids mixing a glTF matrix with animated TRS or losing the inverse 1/32 scale on the player's deployable hardpoint housings.

Repeated positive influence slots for the same joint are coalesced during glTF export. The original joint/byte-weight tuple is retained per affected vertex in mesh extras. This is algebraically equivalent linear skinning and avoids Blender replacing repeated vertex-group weights. The full-library check exposed this on Data_store (12 vertices) and City_Component_MiningSpider01 (4 vertices).

Native vertices are joint-local; glTF skins therefore use identity inverse-bind matrices. This also preserves the blended cable vertices. Blender round-trip checks at start, midpoint and end passed for the throttle, moving monitor, blended cables and autocannon, with maximum bounds error below 2e-7 native units after the interpolation correction.

## Export and remaining limits

```powershell
python tools/bzg-extractor.py models --extract-root work/asset_dump_2026-10-24/raw --model-output-root work/asset_dump_2026-10-24/scenes --model-texture-root work/asset_dump_2026-10-24/textures_png
python -m unittest discover -s tests -v
```

The default model backend is now `scene`. Each glTF references a sibling binary buffer and the PNG texture tree. Keep those folders together. The exporter also builds a standard player-tank inspection assembly. Its native vehicle-picker exterior is calibrated to the gameplay body using matching bounds on all three axes; the observed uniform scale is 32. The gameplay-sized interior attaches to the exterior's native Dome socket with inverse scale 1/32, preserving the cockpit group's native placement.

The installed dataset produced all 1,401 scene resources from 1,401 HSKN candidates, with zero failed models. These counts include repeated assets across archives and variants. All 757 discovered HCAN chunks decoded; this count likewise includes repeats. A finite glTF preview does not recover the invalid UV values described above.

## Blender material inspection

Run Blender 5.2 with the reusable builder after scene, texture and resource extraction:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background --python-exit-code 1 --python tools/build-player-tank-materials.py -- --dump-root work/asset_dump_2026-10-24 --render
```

This writes `scenes/standard_player_tank_materials.blend` with packed images and three inspection cameras (timeline frames 1, 100, 200). Native diffuse RGB is multiplied by the decoded material tint. Alpha roughness has the shader's 0.05 floor. The packed normal RG is decoded as XY*2-1 with reconstructed positive Z and a green-axis conversion for Blender. Native material-model and emission defaults replace the initial guessed metallic/illumination settings; see `RESOURCE_FORMATS.md` for code evidence. Dedicated opacity, height maps and the reflection environment remain connected; parallax is represented as a small bump approximation. Full BRDF, thin-glass/OIT, reflection projection and entity override callbacks remain unresolved or approximated. Blank GUI overlays use their dedicated opacity mask while physical screens retain the blank native render target.

Blender's unit bone-tail parenting under the 32x armature scale displaced socket children. The builder restores native node poses and compensates socket transforms using source world matrices, keeping bone parents and actions intact. Every evaluated render vertex is compared bidirectionally with independently computed glTF skinning before and after saving/reopening. The full assembly is also checked at start/mid/end for throttle, moving monitor (including its screen), and blended cable actions. Results are in `reports/player_tank_materials.json`; the current maximum error is below 1e-7 native units. These checks establish interchange consistency, not running-game socket alias or state validation.

Live GUI render targets are not static textures. Their surfaces, UVs and material references survive, but functioning radar, health, menus and counters require BZR UI logic. Runtime entity states, VFX, shader behaviors, automatic loadout selection and final BZR gameplay integration are not certified by this extraction pass. The assembled cockpit is an inspection scene, not a completed BZR vehicle.

## Categorized Blender libraries

`build-asura-asset-library.py --dump-root PATH --prepare` creates categorized glTF catalogs for all 1,288 render resources. Run the same script in Blender with `--build` to create collection assets in `blender_library/*.blend`. The 113 node-only resources remain in their individual scene exports and the library manifest.

Libraries cover player tanks/skin variants, enemy tank resources, drones/other enemies, turrets, props/structures, weapons, cockpit/displays, debris, and helpers. Counts are native resource/variant counts, not counts of complete unique entities. Bulk libraries reference the shared texture tree with relative paths. Frame 0 shows the gallery; frames 1..N frame individual resources. Internal transforms, skinning, markers and candidate clips remain intact; only the outer Gallery root translation changes for inspection layout. Collections are marked as Blender assets.

`assembled_examples.blend` adds 12 explicit inspection assemblies: three player exteriors with the common cockpit, four enemy cannon-tank examples, UFO, scout bike, hopper, droid and a cannon turret. Picker body scale is checked against native gameplay bounds. Socket transforms are native; aliases selecting which interchangeable model occupies each socket are recorded and remain unverified against native entity state/loadout code. These examples must not be presented as decoded stock loadouts. Full resource provenance, attachments, categories and remaining limits live in `blender_library/manifest.json`; per-batch reload/vertex checks are saved beside each library.

`player_weapon_assemblies.blend` adds 15 examples for the exterior weapons that swap on the tank: 11 gun families, three hardpoint housings, and a full player tank with an inspection loadout. Separate guided-launcher rings and node-only muzzle controllers attach at native sockets. The 119-resource inventory includes 112 visible resources plus seven node-only resources, including variants, ammunition and interfaces; it is not a count of distinct guns. All 39 library files passed saved/reopened vertex checks. Weapon motion validation covers 39 clips at start/mid/end (117 samples), including fire, reload, stow, charge, launcher rings and hardpoint deployment. Entity pairing, interpolation and gameplay remain subject to the runtime limits above.
