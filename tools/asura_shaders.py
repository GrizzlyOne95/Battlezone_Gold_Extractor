"""Bounded DXBC/bank readers and Windows compiler reflection.

Uses the documented D3DDisassemble/D3DReflect ABI from Microsoft's Windows SDK.
Bytecode is preserved; disassembly is not original HLSL source.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
import hashlib
import os
import struct
import uuid


def u32(data, offset):
    if offset < 0 or offset + 4 > len(data):
        raise ValueError(f'Truncated u32 at {offset}')
    return struct.unpack_from('<I', data, offset)[0]


def read_dxbc(data: bytes, offset=0) -> dict:
    if data[offset:offset+4] != b'DXBC' or offset < 0 or offset+32 > len(data):
        raise ValueError('Missing or truncated DXBC header')
    version, size, count = struct.unpack_from('<III', data, offset+20)
    if version != 1 or count > 1024 or size < 32+count*4 or offset+size > len(data):
        raise ValueError('Invalid DXBC container bounds')
    chunks=[];ranges=[];profile=None
    for index in range(count):
        relative=u32(data,offset+32+index*4)
        if relative<32+count*4 or relative+8>size:
            raise ValueError('Invalid DXBC chunk offset')
        length=u32(data,offset+relative+4)
        if relative+8+length>size:
            raise ValueError('DXBC chunk overruns container')
        tag=data[offset+relative:offset+relative+4].decode('ascii')
        if any(relative < end and relative+8+length > start for start,end in ranges):
            raise ValueError('Overlapping DXBC chunks')
        ranges.append((relative,relative+8+length))
        chunks.append(dict(tag=tag,offset=relative+8,size=length))
        if tag in ('SHDR','SHEX','DXIL') and length>=4:
            token=u32(data,offset+relative+8)
            stage={0:'ps',1:'vs',2:'gs',3:'hs',4:'ds',5:'cs'}.get(token>>16,'unknown')
            profile=f'{stage}_{(token>>4)&15}_{token&15}'
    payload=data[offset:offset+size]
    return dict(size=size,profile=profile,chunks=chunks,sha256=hashlib.sha256(payload).hexdigest(),checksum=payload[4:20].hex())


@dataclass
class ShaderEntry:
    index:int
    offset:int
    permutation:int|None
    metadata:dict
    bytecode:bytes


def read_shader_bank(data:bytes, particle=False) -> tuple[list[ShaderEntry],bytes]:
    count=u32(data,0)
    if count>100000:
        raise ValueError('Invalid shader bank count')
    cursor=4;entries=[]
    for index in range(count):
        permutation=None
        if not particle:
            if cursor+8>len(data):raise ValueError('Truncated permutation key')
            permutation=struct.unpack_from('<Q',data,cursor)[0];cursor+=8
        size=u32(data,cursor);cursor+=4
        meta=read_dxbc(data,cursor)
        if size!=meta['size']:raise ValueError('Bank size differs from DXBC size')
        entries.append(ShaderEntry(index,cursor,permutation,meta,data[cursor:cursor+size]))
        cursor+=size
    tail=data[cursor:]
    if not particle and tail:raise ValueError('Unexpected model shader-bank tail')
    # The particle bank has one trailing u32 per program. Preserve even if nonzero.
    if particle and len(tail) not in (0,count*4):raise ValueError('Unknown particle shader-bank tail layout')
    return entries,tail


class ShaderDesc(C.Structure):
    _fields_=[('Version',C.c_uint32),('Creator',C.c_char_p)]+[(n,C.c_uint32) for n in ('Flags ConstantBuffers BoundResources InputParameters OutputParameters InstructionCount TempRegisterCount TempArrayCount DefCount DclCount TextureNormalInstructions TextureLoadInstructions TextureCompInstructions TextureBiasInstructions TextureGradientInstructions FloatInstructionCount IntInstructionCount UintInstructionCount StaticFlowControlCount DynamicFlowControlCount MacroInstructionCount ArrayInstructionCount CutInstructionCount EmitInstructionCount GSOutputTopology GSMaxOutputVertexCount InputPrimitive PatchConstantParameters cGSInstanceCount cControlPoints HSOutputPrimitive HSPartitioning TessellatorDomain cBarrierInstructions cInterlockedInstructions cTextureStoreInstructions').split()]


class BufferDesc(C.Structure):
    _fields_=[('Name',C.c_char_p)]+[(n,C.c_uint32) for n in ('Type Variables Size uFlags').split()]


class VariableDesc(C.Structure):
    _fields_=[('Name',C.c_char_p)]+[(n,C.c_uint32) for n in ('StartOffset Size uFlags').split()]+[('DefaultValue',C.c_void_p)]+[(n,C.c_uint32) for n in ('StartTexture TextureSize StartSampler SamplerSize').split()]


class TypeDesc(C.Structure):
    _fields_=[(n,C.c_uint32) for n in ('Class Type Rows Columns Elements Members Offset').split()]+[('Name',C.c_char_p)]


class BindDesc(C.Structure):
    _fields_=[('Name',C.c_char_p)]+[(n,C.c_uint32) for n in ('Type BindPoint BindCount uFlags ReturnType Dimension NumSamples').split()]


class SignatureDesc(C.Structure):
    _fields_=[('SemanticName',C.c_char_p)]+[(n,C.c_uint32) for n in ('SemanticIndex Register SystemValueType ComponentType').split()]+[('Mask',C.c_uint8),('ReadWriteMask',C.c_uint8),('Stream',C.c_uint32),('MinPrecision',C.c_uint32)]


def describe(value):
    output={}
    for name,_ in value._fields_:
        item=getattr(value,name)
        if isinstance(item,bytes):item=item.decode('utf-8','replace')
        if name!='DefaultValue':output[name]=item
    return output


class WindowsCompiler:
    def __init__(self):
        if os.name!='nt':raise OSError('D3D compiler reflection requires Windows')
        self.dll=C.WinDLL('d3dcompiler_47.dll')
        self.dll.D3DDisassemble.argtypes=[C.c_void_p,C.c_size_t,C.c_uint32,C.c_char_p,C.POINTER(C.c_void_p)]
        self.dll.D3DDisassemble.restype=C.c_int32
        self.dll.D3DReflect.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.POINTER(C.c_void_p)]
        self.dll.D3DReflect.restype=C.c_int32

    @staticmethod
    def method(pointer,index,result,*args):
        vtable=C.cast(pointer,C.POINTER(C.POINTER(C.c_void_p))).contents
        return C.WINFUNCTYPE(result,C.c_void_p,*args)(vtable[index])

    def call(self,pointer,index,result,args,values):
        return self.method(pointer,index,result,*args)(pointer,*values)

    @staticmethod
    def check(status):
        if status<0:raise ValueError(f'D3D compiler HRESULT 0x{status&0xffffffff:08x}')

    def disassemble(self,bytecode):
        source=C.create_string_buffer(bytecode);blob=C.c_void_p()
        self.check(self.dll.D3DDisassemble(source,len(bytecode),2|4,None,C.byref(blob)))
        try:
            pointer=self.call(blob,3,C.c_void_p,[],[])
            size=self.call(blob,4,C.c_size_t,[],[])
            return C.string_at(pointer,size).rstrip(b'\0').decode('utf-8','replace')
        finally:self.call(blob,2,C.c_uint32,[],[])

    def type_info(self,pointer,depth=0):
        if not pointer or depth>12:raise ValueError('Invalid shader type recursion')
        desc=TypeDesc()
        self.check(self.call(pointer,0,C.c_int32,[C.POINTER(TypeDesc)],[C.byref(desc)]))
        data=describe(desc)
        if desc.Members>1024:raise ValueError('Invalid shader member count')
        if desc.Members:
            data['member_types']=[]
            for i in range(desc.Members):
                member=self.call(pointer,1,C.c_void_p,[C.c_uint32],[i])
                name=self.call(pointer,3,C.c_char_p,[C.c_uint32],[i])
                data['member_types'].append(dict(name=name.decode('utf-8','replace') if name else None,type=self.type_info(member,depth+1)))
        return data

    def reflect(self,bytecode):
        source=C.create_string_buffer(bytecode);pointer=C.c_void_p()
        iid=C.create_string_buffer(uuid.UUID('8d536ca1-0cca-4956-a837-786963755584').bytes_le)
        self.check(self.dll.D3DReflect(source,len(bytecode),iid,C.byref(pointer)))
        try:
            desc=ShaderDesc()
            self.check(self.call(pointer,3,C.c_int32,[C.POINTER(ShaderDesc)],[C.byref(desc)]))
            if max(desc.ConstantBuffers,desc.BoundResources,desc.InputParameters,desc.OutputParameters)>10000:raise ValueError('Invalid reflection counts')
            data=dict(shader=describe(desc),resources=[],constant_buffers=[],inputs=[],outputs=[])
            for i in range(desc.BoundResources):
                bind=BindDesc()
                self.check(self.call(pointer,6,C.c_int32,[C.c_uint32,C.POINTER(BindDesc)],[i,C.byref(bind)]))
                data['resources'].append(describe(bind))
            for i in range(desc.ConstantBuffers):
                buffer=self.call(pointer,4,C.c_void_p,[C.c_uint32],[i]);bd=BufferDesc()
                self.check(self.call(buffer,0,C.c_int32,[C.POINTER(BufferDesc)],[C.byref(bd)]))
                record=describe(bd);record['variables']=[]
                for vi in range(bd.Variables):
                    variable=self.call(buffer,1,C.c_void_p,[C.c_uint32],[vi]);vd=VariableDesc()
                    self.check(self.call(variable,0,C.c_int32,[C.POINTER(VariableDesc)],[C.byref(vd)]))
                    value=describe(vd)
                    if vd.DefaultValue:value['default_hex']=C.string_at(vd.DefaultValue,vd.Size).hex()
                    type_ptr=self.call(variable,1,C.c_void_p,[],[])
                    value['type']=self.type_info(type_ptr)
                    record['variables'].append(value)
                data['constant_buffers'].append(record)
            for method,count,key in [(7,desc.InputParameters,'inputs'),(8,desc.OutputParameters,'outputs')]:
                for i in range(count):
                    signature=SignatureDesc()
                    self.check(self.call(pointer,method,C.c_int32,[C.c_uint32,C.POINTER(SignatureDesc)],[i,C.byref(signature)]))
                    data[key].append(describe(signature))
            return data
        finally:self.call(pointer,2,C.c_uint32,[],[])
