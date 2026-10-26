"""Structured Battlezone Gold scene readers. Offsets are checked, never guessed.

Render resource v6: counts + submeshes + quantization transform + 40-byte vertices.
HSKN: parent table, local TR quaternions, named geometry nodes, auxiliary data.
HCAN: per-node key spans, quantized rotations/times, translations, node hash table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import numpy as np


def name_hash(name: str) -> int:
    value = 0
    for char in name.lower():
        value = (31 * value + ord(char)) & 0xffffffff
    return value


def sample_rotations(times: np.ndarray, values: np.ndarray, fps: int = 60):
    """Keep every native key and add slerp samples for Blender's linear F-curves."""
    if len(times) < 2 or times[-1] <= times[0]:
        return times, values
    values = values.astype(np.float64).copy()
    for i in range(1, len(values)):
        if np.dot(values[i - 1], values[i]) < 0:
            values[i] *= -1
    extra = np.arange(float(times[0]), float(times[-1]), 1 / fps)
    dense_times = np.unique(np.r_[times, extra].astype(np.float32))
    interval = np.clip(np.searchsorted(times, dense_times, side='right') - 1, 0, len(times) - 2)
    fraction = ((dense_times - times[interval]) / (times[interval + 1] - times[interval])).astype(np.float64)
    left, right = values[interval], values[interval + 1]
    dot = np.clip(np.sum(left * right, axis=1), -1, 1)
    theta = np.arccos(dot)
    sine = np.sin(theta)
    near = np.abs(sine) < 1e-6
    sine[near] = 1
    a = np.sin((1 - fraction) * theta) / sine
    b = np.sin(fraction * theta) / sine
    a[near], b[near] = 1 - fraction[near], fraction[near]
    result = left * a[:, None] + right * b[:, None]
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return dense_times, result.astype(np.float32)


def padded_name(blob: bytes, offset: int) -> tuple[str, int]:
    end = blob.index(0, offset)
    name = blob[offset:end].decode('ascii')
    if not name or any(ord(c) < 32 for c in name):
        raise ValueError('Invalid resource name')
    return name, offset + ((end - offset + 4) // 4) * 4


@dataclass
class NativeNode:
    name: str
    parent: int
    translation: np.ndarray
    rotation: np.ndarray
    bounds: np.ndarray | None
    cpu_positions: np.ndarray | None
    cpu_faces: np.ndarray | None
    geometry_offset: int
    geometry_size: int
    native_flags: int = 0


@dataclass
class Skeleton:
    name: str
    nodes: list[NativeNode]
    tail_offset: int
    flags: int


def read_skeleton(blob: bytes) -> Skeleton:
    flags, count = struct.unpack_from('<II', blob)
    if not 1 <= count <= 8192:
        raise ValueError('Invalid node count')
    name, cursor = padded_name(blob, 8)
    parents = struct.unpack_from('<' + 'I' * count, blob, cursor)
    cursor += count * 4
    transforms = np.frombuffer(blob, dtype='<f4', offset=cursor, count=count * 7).reshape(count, 7).copy()
    if not np.isfinite(transforms).all():
        raise ValueError('Nonfinite node transform')
    for index, parent in enumerate(parents):
        if (index == 0 and parent not in (0, 0xffffffff)) or (index > 0 and parent >= index):
            raise ValueError('Invalid or cyclic parent table')
    if not np.allclose(np.linalg.norm(transforms[:, 3:], axis=1), 1, atol=0.001):
        raise ValueError('Invalid bind rotation')
    cursor += count * 28
    has_geometry = blob[cursor]
    if has_geometry not in (0, 1):
        raise ValueError('Unknown HSKN geometry flag')
    cursor += 1
    nodes = []
    for index in range(count):
        node_name, cursor = padded_name(blob, cursor)
        node_flag = blob[cursor]
        if node_flag not in (0, 1):
            raise ValueError(f'Unsupported node flag {node_flag}')
        cursor += 1
        size = struct.unpack_from('<I', blob, cursor)[0]
        end = cursor + 4 + size
        if end > len(blob):
            raise ValueError('Node geometry overruns chunk')
        bounds = positions = faces = None
        if size:
            version, nv, nf = struct.unpack_from('<III', blob, cursor + 4)
            if version != 14 or nv < 1 or nf < 1:
                raise ValueError('Unsupported node geometry layout')
            # 36-byte header, six bounds floats, radius float, then XYZ triplets.
            bounds = np.array(struct.unpack_from('<6f', blob, cursor + 36)).reshape(3, 2)
            vertex_offset = cursor + 64
            index_offset = vertex_offset + nv * 12
            if index_offset + nf * 8 > end:
                raise ValueError('Invalid polygon table bounds')
            positions = np.frombuffer(blob, dtype='<f4', offset=vertex_offset, count=nv * 3).reshape(nv, 3).copy()
            faces = np.frombuffer(blob, dtype='<u2', offset=index_offset, count=nf * 4).reshape(nf, 4).copy()
            if not np.isfinite(positions).all() or np.any(faces[:, :3] >= nv) or np.any((faces[:, 3] != 65535) & (faces[:, 3] >= nv)):
                raise ValueError('Invalid native polygon indices')
            if np.any(positions.min(0) < bounds[:, 0] - 0.00002) or np.any(positions.max(0) > bounds[:, 1] + 0.00002):
                raise ValueError('Native positions lie outside stored bounds')
        nodes.append(NativeNode(node_name, -1 if index == 0 else parents[index], transforms[index, :3], transforms[index, 3:], bounds, positions, faces, cursor, size, node_flag))
        cursor = end
    return Skeleton(name, nodes, cursor, flags)


@dataclass
class RenderMesh:
    positions: np.ndarray
    normals: np.ndarray
    uv0: np.ndarray
    uv1: np.ndarray
    joints: np.ndarray
    weights: np.ndarray
    triangles: np.ndarray
    parts: list[dict]
    quantization_scale: np.ndarray
    quantization_offset: np.ndarray
    bone_palettes: list | None = None
    nonfinite_uv: list | None = None


def read_render_mesh(blob: bytes) -> RenderMesh:
    count, nv, ni, nt = struct.unpack_from('<4I', blob)
    if len(blob) == 20 and (count, nv, ni, nt) == (0, 0, 0, 0):
        return RenderMesh(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 4), dtype=np.uint8), np.empty((0, 4)), np.empty((0, 3), dtype=np.uint16), [], np.ones(3), np.zeros(3))
    start = 44 + count * 24
    index_end = start + nv * 40 + ni * 2
    if not 1 <= count <= 2048 or nv < 1 or ni != nt * 3 or index_end > len(blob):
        raise ValueError('Unsupported render resource layout')
    parts = []
    cursor = 0
    for index in range(count):
        material, zero, num_indices, zero2, flag, palette = struct.unpack_from('<6I', blob, 20 + index * 24)
        if num_indices % 3:
            raise ValueError('Invalid render submesh')
        parts.append(dict(material_hash=material, first_triangle=cursor // 3, triangle_count=num_indices // 3, flags=flag, bone_palette=palette, unknown=[zero, zero2]))
        cursor += num_indices
    if cursor != ni:
        raise ValueError('Submesh counts do not match index buffer')
    palette_count = struct.unpack_from('<I', blob, 16)[0]
    if palette_count > 8192:
        raise ValueError('Invalid bone palette count')
    palettes = []
    cursor = index_end
    for index in range(palette_count):
        if cursor + 4 > len(blob):
            raise ValueError('Truncated bone palette size')
        length = struct.unpack_from('<I', blob, cursor)[0]
        cursor += 4
        if length > 8192 or cursor + length * 4 > len(blob):
            raise ValueError('Invalid bone palette bounds')
        palettes.append(list(struct.unpack_from('<' + 'I' * length, blob, cursor)))
        cursor += length * 4
    if cursor != len(blob):
        raise ValueError('Unparsed render resource tail')
    if palette_count and any(part['bone_palette'] >= palette_count for part in parts):
        raise ValueError('Invalid submesh bone palette reference')
    quant = np.array(struct.unpack_from('<6f', blob, 20 + count * 24))
    if not np.isfinite(quant).all():
        raise ValueError('Nonfinite quantization transform')
    raw = np.frombuffer(blob, dtype=np.uint8, offset=start, count=nv * 40).reshape(nv, 40)
    positions = raw[:, :6].copy().view('<u2').reshape(nv, 3).astype(np.float64) / 65536 * quant[:3] + quant[3:]
    packed_normals = raw[:, 8:12].copy().view('<u4').reshape(nv)
    normals = np.stack([((packed_normals >> shift) & 1023).astype(np.float32) / 1023 * 2 - 1 for shift in (0, 10, 20)], axis=1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    uv0 = raw[:, 24:28].copy().view('<f2').reshape(nv, 2).astype(np.float32)
    uv1 = raw[:, 28:32].copy().view('<f2').reshape(nv, 2).astype(np.float32)
    weights = raw[:, 32:36].astype(np.float32) / 255
    joints = raw[:, 36:40].copy()
    if not np.allclose(weights.sum(1), 1):
        raise ValueError('Invalid skin weights')
    nonfinite_uv = []
    for channel, uv in enumerate((uv0, uv1)):
        for vertex, component in np.argwhere(~np.isfinite(uv)):
            bits = int(raw[vertex, 24 + channel * 4:28 + channel * 4].copy().view('<u2')[component])
            nonfinite_uv.append(dict(channel=channel, vertex=int(vertex), component=int(component), native_f16_bits=f'{bits:04x}', interchange_value=0.0))
        # JSON/glTF cannot represent infinity. Keep exact source bits above and
        # use a declared finite preview value only in the interchange UV buffer.
        uv[~np.isfinite(uv)] = 0
    triangles = np.frombuffer(blob, dtype='<u2', offset=start + nv * 40, count=ni).reshape(nt, 3).copy()
    if triangles.max() >= nv:
        raise ValueError('Render index outside vertex buffer')
    return RenderMesh(positions, normals, uv0, uv1, joints, weights, triangles, parts, quant[:3], quant[3:], palettes, nonfinite_uv)


def coalesce_skin_influences(joints,weights):
    """Combine repeated joint slots without changing native linear skinning.

    Blender's glTF vertex-group import replaces repeated positive weights for
    the same joint. Keep the exact original byte influences in mesh extras.
    """
    joints=np.asarray(joints,dtype=np.uint8).copy();source=np.asarray(weights,dtype=np.float32);result=source.copy();changed=set()
    for slot in range(1,4):
        for previous in range(slot):
            mask=(joints[:,slot]==joints[:,previous])&(result[:,slot]>0)
            changed.update(np.flatnonzero(mask).tolist())
            result[mask,previous]+=result[mask,slot];result[mask,slot]=0
    records=[dict(vertex=i,native_joints=joints[i].tolist(),native_weight_bytes=np.rint(source[i]*255).astype(int).tolist()) for i in sorted(changed)]
    return joints,result,records


def validate_render_binding(mesh: RenderMesh, skeleton: Skeleton) -> dict:
    used = mesh.weights > 0
    if np.any(mesh.joints[used] >= len(skeleton.nodes)):
        raise ValueError('Render joint is absent from native skeleton')
    if any(index >= len(skeleton.nodes) for palette in (mesh.bone_palettes or []) for index in palette):
        raise ValueError('Bone palette references absent native node')
    checks = []
    for index, node in enumerate(skeleton.nodes):
        mask = (mesh.joints[:, 0] == index) & (mesh.weights[:, 0] == 1)
        if node.bounds is None or not mask.any():
            continue
        actual = np.stack([mesh.positions[mask].min(0), mesh.positions[mask].max(0)], axis=1)
        error = float(np.max(np.abs(actual - node.bounds)))
        # CPU meshes can omit triangles such as transparent/decorative geometry.
        checks.append(dict(node=index, name=node.name, max_bounds_error=error, within_quantization=error <= float(np.max(np.abs(mesh.quantization_scale))) / 65536 * 2 + 1e-6))
    return dict(nodes=len(skeleton.nodes), vertices=len(mesh.positions), triangles=len(mesh.triangles), blended_vertices=int(np.sum(np.count_nonzero(mesh.weights, axis=1) > 1)), bounds_checks=checks,nonfinite_source_uv_components=mesh.nonfinite_uv or [])


@dataclass
class Animation:
    name: str
    duration: float
    tracks: np.ndarray
    rotations: np.ndarray
    rotation_times: np.ndarray
    translations: np.ndarray
    translation_times: np.ndarray
    node_hashes: tuple[int, ...]
    raw_tail_offset: int
    root_motion_track: tuple[int, ...] | None
    motion_parameter: float


def read_animation(blob: bytes) -> Animation:
    count, duration, zero, nq, nt, flags = struct.unpack_from('<IfIIII', blob)
    if not 1 <= count <= 8192 or not np.isfinite(duration) or duration < 0 or flags != 0:
        raise ValueError('Unsupported HCAN header')
    name, cursor = padded_name(blob, 24)
    tracks = np.zeros((count, 4), dtype=np.uint32)
    for i in range(count):
        tracks[i] = struct.unpack_from('<HHII', blob, cursor + i * 12)
    cursor += count * 12
    root_motion = None
    if int(tracks[:, 0].sum()) != nq or int(tracks[:, 1].sum()) != nt:
        # A separate root-motion track follows named-node tracks when present.
        root_motion = struct.unpack_from('<HHII', blob, cursor)
        cursor += 12
    all_tracks = np.vstack([tracks, root_motion]) if root_motion is not None else tracks
    if int(all_tracks[:, 0].sum()) != nq or int(all_tracks[:, 1].sum()) != nt:
        raise ValueError('Animation track totals do not match header')
    if np.any(tracks[:, 0] + tracks[:, 2] > nq) or np.any(tracks[:, 1] + tracks[:, 3] > nt):
        raise ValueError('Invalid animation key spans')
    q = np.frombuffer(blob, dtype='<u2', offset=cursor, count=nq * 4).reshape(nq, 4).astype(np.float64)
    q[:, :3] = (q[:, :3] - 32767) / 32767
    q[:, 3] /= 65535
    cursor += nq * 8
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if not np.allclose(norms, 1, atol=0.0001):
        raise ValueError('Invalid decoded animation quaternion')
    q /= norms
    qt = np.frombuffer(blob, dtype='<u2', offset=cursor, count=nq).astype(np.float64) / 65535 * duration
    cursor += nq * 2
    t = np.frombuffer(blob, dtype='<f4', offset=cursor, count=nt * 3).reshape(nt, 3).copy()
    cursor += nt * 12
    tt = np.frombuffer(blob, dtype='<u2', offset=cursor, count=nt).astype(np.float64) / 65535 * duration
    cursor += nt * 2
    if not np.isfinite(t).all():
        raise ValueError('Invalid animation translation')
    for nr, np_, ir, ip in all_tracks:
        if np.any(np.diff(qt[ir:ir + nr]) < 0) or np.any(np.diff(tt[ip:ip + np_]) < 0):
            raise ValueError('Nonmonotonic animation times')
    # Trailing hashes follow an animation/bounds hash, before the event table.
    # Only the zero-event layout is supported; retain the entire source in all cases.
    if struct.unpack_from('<II', blob, len(blob) - 8) != (1, 0):
        raise ValueError('Unsupported HCAN event footer')
    hash_offset = len(blob) - 8 - count * 4
    if hash_offset < cursor:
        raise ValueError('Animation hash table overlaps keys')
    hashes = struct.unpack_from('<' + 'I' * count, blob, hash_offset)
    return Animation(name, duration, tracks, q.astype(np.float32), qt.astype(np.float32), t, tt.astype(np.float32), hashes, cursor, root_motion, struct.unpack_from('<f', blob, 8)[0])


def read_markers(blob: bytes) -> tuple[str, list[dict]]:
    count = struct.unpack_from('<I', blob)[0]
    owner, cursor = padded_name(blob, 4)
    markers = []
    for index in range(count):
        values = struct.unpack_from('<12fI', blob, cursor)
        cursor += 52
        name, cursor = padded_name(blob, cursor)
        markers.append(dict(name=name, translation=list(values[:3]), basis=np.array(values[3:12]).reshape(3, 3).tolist(), parent=values[12]))
    if cursor != len(blob):
        raise ValueError('Unparsed attachment data')
    return owner, markers


def read_materials(blob: bytes) -> dict[int, dict]:
    from asura_materials import read_material_table
    return read_material_table(blob)['materials']
