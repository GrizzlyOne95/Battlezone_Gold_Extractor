#!/usr/bin/env python3
"""Create PCM16 WAV copies while retaining native audio unchanged."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import shutil
import subprocess
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise SystemExit('FFmpeg is required')
    sources = sorted(p for p in args.input_root.rglob('*') if p.is_file())
    def convert(src):
        relative = src.relative_to(args.input_root)
        dst = args.output_root / relative.with_suffix('.wav')
        dst.parent.mkdir(parents=True,exist_ok=True)
        proc = subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-y','-i',str(src),'-map','0:a:0','-c:a','pcm_s16le',str(dst)],capture_output=True,text=True,encoding='utf-8',errors='replace')
        row = dict(source=str(src),destination=str(dst),status='failed',channels='',sample_rate='',frames='',duration='',error=proc.stderr.strip())
        if proc.returncode == 0:
            with wave.open(str(dst),'rb') as wav:
                if wav.getsampwidth() != 2 or wav.getcomptype() != 'NONE':
                    row['error'] = 'Output is not PCM16'
                else:
                    row.update(status='ok',channels=wav.getnchannels(),sample_rate=wav.getframerate(),frames=wav.getnframes(),duration=wav.getnframes()/wav.getframerate())
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(convert,sources))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0] if rows else ['source','status']);writer.writeheader();writer.writerows(rows)
    failures=sum(r['status']!='ok' for r in rows)
    print(f'PCM audio: {len(rows)-failures} OK, {failures} failed',flush=True)
    return int(failures>0)


if __name__=='__main__':
    raise SystemExit(main())
