import struct, numpy as np

path = r'C:\Users\istuart\Documents\GIT\BattlezoneGold_Extract\work\extracted\full_all\chars__actors_pc\HSKN_chunk\.5564.dat'
blob = open(path, 'rb').read()

# The OBJ vertices definitely don't come from .5564.dat's binary at these offsets.
# Let me check if the "hopper_tests_v8_blocks" came from a DIFFERENT source file -
# maybe an extensionless model file rather than the HSKN_chunk

# Search for the extensionless hopper model file
import os
for root, dirs, files in os.walk(r'C:\Users\istuart\Documents\GIT\BattlezoneGold_Extract\work\extracted\full_all'):
    if 'HSKN_chunk' in root:
        continue
    for f in files:
        fpath = os.path.join(root, f)
        try:
            blob2 = open(fpath, 'rb').read()
            if len(blob2) < 100:
                continue
            name = blob2[:64].split(b'\x00')[0].decode('ascii', errors='ignore')
            if 'opper' in name or 'OPPER' in name:
                print(f'Found: {fpath} ({len(blob2)} bytes)')
                print(f'  Name: {name[:50]}')
        except:
            pass

# Also check: maybe the "Hopper" extensionless file is the source
# The older script export-model-obj-candidates.py reads extensionless files
# Let's look for it directly
for root, dirs, files in os.walk(r'C:\Users\istuart\Documents\GIT\BattlezoneGold_Extract\work\extracted\full_all\chars__actors_pc'):
    for f in files:
        if not f.startswith('.') and not f.endswith('.dat') and not f.endswith('.wav') and not f.endswith('.tga') and not f.endswith('.dds') and not f.endswith('.bmp'):
            fpath = os.path.join(root, f)
            if os.path.isfile(fpath):
                sz = os.path.getsize(fpath)
                if sz > 5000:
                    try:
                        b = open(fpath, 'rb').read(64)
                        name = b.split(b'\x00')[0].decode('ascii', errors='ignore')
                        if 'opper' in name.lower():
                            print(f'{sz}: {fpath}  [{name[:40]}]')
                    except:
                        pass
