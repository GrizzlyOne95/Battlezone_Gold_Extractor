"""Regression checks for archive coverage and lossless asset discovery."""
import csv
import importlib.util
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cli = load('bzg_cli', 'bzg-extractor.py')
native = load('asura_native', 'extract-asura-native.py')


def resource(name, data):
    name = name.encode() + b'\0'
    name += bytes((-len(name)) % 4)
    body = struct.pack('<III', 0, 0, len(data)) + name + data
    return b'RSCF' + struct.pack('<III', len(body) + 16, 0, 0) + body


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def sniff(self, data, fn):
        p = self.root / 'payload.dat'
        p.write_bytes(data)
        return fn(p)

    def test_normalized_extensions_do_not_collide(self):
        self.assertEqual(cli.rel_with_extension(Path('a.tga'), '.dds'), Path('a.tga.dds'))
        self.assertNotEqual(cli.rel_with_extension(Path('a.tga'), '.dds'), cli.rel_with_extension(Path('a.dds'), '.dds'))
        self.assertEqual(cli.rel_with_extension(Path('.1.dat'), '.wav'), Path('.1.dat.wav'))

    def test_binary_sync_is_not_mp3(self):
        self.assertIsNone(self.sniff(b'\xff\xff\0\0' + bytes(600), cli.sniff_audio_extension))
        self.assertIsNone(self.sniff(b'\xff\xfb\x90\0' + bytes(600), cli.sniff_audio_extension))

    def test_two_mp3_frames(self):
        frame = b'\xff\xfb\x90\0' + bytes(413)
        self.assertEqual(self.sniff(frame * 2, cli.sniff_audio_extension), '.mp3')

    def test_riff_requires_wave_form_type(self):
        self.assertIsNone(self.sniff(b'RIFF' + bytes(4) + b'AVI WAVE', cli.sniff_audio_extension))
        self.assertEqual(self.sniff(b'RIFF' + bytes(4) + b'WAVE', cli.sniff_audio_extension), '.wav')

    def test_misnamed_dds(self):
        self.assertEqual(self.sniff(b'DDS ' + bytes(124), cli.sniff_texture_extension), '.dds')

    def test_bc7_texture_fallback_writes_valid_png(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow not installed')
        header = bytearray(148)
        header[:4] = b'DDS '
        struct.pack_into('<7I', header, 4, 124, 0x81007, 4, 4, 16, 0, 1)
        struct.pack_into('<II4s', header, 76, 32, 4, b'DX10')
        struct.pack_into('<I', header, 108, 0x1000)
        struct.pack_into('<5I', header, 128, 99, 3, 0, 1, 0)
        (self.root / 'bc7.dds').write_bytes(header + b'\x40' + bytes(15))
        report = self.root / 'conversion.csv'
        with report.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['SourcePath', 'DestPath', 'Status', 'ExitCode', 'Converter'])
            writer.writeheader(); writer.writerow(dict(SourcePath='bc7.dds', DestPath='bc7.png', Status='Failed', ExitCode='1', Converter='none'))
        self.assertTrue(cli.repair_texture_report(self.root, self.root, report))
        with Image.open(self.root / 'bc7.png') as img:
            self.assertEqual(img.size, (4, 4))
            img.verify()

    def test_empty_and_repeated_resources_preserved(self):
        payload = b'Asura   ' + resource('same.wav', b'first') + resource('same.wav', b'second') + resource('empty', b'')
        summary = native.ExtractSummary('', '', '', '')
        native.extract_archive_payload(payload, 0, self.root, True, summary)
        self.assertEqual((self.root / 'same.wav').read_bytes(), b'first')
        self.assertEqual((self.root / 'same__duplicate_2.wav').read_bytes(), b'second')
        self.assertEqual((self.root / 'empty').read_bytes(), b'')
        self.assertFalse(summary.warnings)

    def test_empty_chunk_does_not_stop_next_resource(self):
        empty = b'TEST' + struct.pack('<III', 16, 0, 0)
        payload = b'Asura   ' + empty + resource('later', b'good')
        summary = native.ExtractSummary('', '', '', '')
        native.extract_archive_payload(payload, 0, self.root, True, summary)
        self.assertEqual((self.root / 'later').read_bytes(), b'good')
        self.assertFalse(summary.warnings)

    def test_zero_terminator_and_corrupt_tail_are_distinguished(self):
        blob = b'Asura   ' + resource('x', b'y')
        summary = native.ExtractSummary('', '', '', '')
        native.extract_archive_payload(blob + bytes(16), 0, self.root, True, summary)
        self.assertFalse(summary.warnings)
        summary = native.ExtractSummary('', '', '', '')
        native.extract_archive_payload(blob + b'BAD', 0, self.root, True, summary)
        self.assertTrue(summary.warnings)

    def test_zlib_and_chunked_wrapper(self):
        payload = b'Asura   ' + resource('x', b'y')
        compressed = zlib.compress(payload)
        single = b'AsuraZlb' + bytes(12) + compressed
        self.assertEqual(native.unwrap_input(single)[:2], (payload, 0))
        chunked = b'AsuraZbb' + struct.pack('<II', len(compressed), len(payload)) + struct.pack('<II', len(compressed), len(payload)) + compressed
        self.assertEqual(native.unwrap_input(chunked)[:2], (payload, 0))
        with self.assertRaises(native.ParseError):
            native.unwrap_input(chunked[:-2])
        wrong_size = bytearray(chunked)
        struct.pack_into('<I', wrong_size, 12, len(payload) + 1)
        with self.assertRaises(native.ParseError):
            native.unwrap_input(bytes(wrong_size))

    def test_default_is_full_coverage(self):
        args = cli.build_parser().parse_args(['extract'])
        self.assertEqual(args.target_set, 'AllAsuraHeaders')
        combined = cli.build_parser().parse_args(['run'])
        self.assertIn('resources', cli.parse_task_list(combined.tasks))
        resources = cli.build_parser().parse_args(['resources','--resource-output-root','output/resources'])
        self.assertEqual(resources.resource_output_root, 'output/resources')

    @unittest.skipUnless(shutil.which('powershell') and shutil.which('ffmpeg'), 'Texture conversion runtime')
    def test_texture_pipeline_copies_png_and_avoids_stem_collision(self):
        import binascii
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', binascii.crc32(kind + data) & 0xffffffff)
        png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(b'\0\xff\0\0')) + chunk(b'IEND', b'')
        source = self.root / 'raw'
        source.mkdir()
        (source / 'same.png').write_bytes(png)
        # PNG content deliberately mislabeled .dat must still be found and copied.
        (source / 'same.dat').write_bytes(png)
        report = self.root / 'conversion.csv'
        cmd = [sys.executable, str(ROOT / 'tools/bzg-extractor.py'), 'textures', '--extract-root', str(source), '--texture-stage-root', str(self.root / 'stage'), '--texture-output-root', str(self.root / 'png'), '--texture-report-path', str(report), '--texture-discovery-report', str(self.root / 'discovery.csv'), '--no-skip-existing']
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        outputs = list((self.root / 'png').rglob('*.png'))
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(p.read_bytes() == png for p in outputs))
        with report.open(encoding='utf-8-sig', newline='') as f:
            self.assertTrue(all(row['Status'] == 'OK' for row in csv.DictReader(f)))

    @unittest.skipUnless(shutil.which('powershell'), 'Windows PowerShell integration')
    def test_batch_covers_asr_gui_locale_and_pointers(self):
        game = self.root / 'game'
        (game / 'envs').mkdir(parents=True)
        blob = b'Asura   ' + resource('asset', b'payload')
        for name in ['x.pc', 'x.asr_en', 'x.gui', 'x.en', 'envs/x.pc_entdata']:
            (game / name).write_bytes(blob)
        (game / 'envs/x.pc_entheader').write_text('envs\\x.pc_entdata')
        (game / 'unrelated.txt').write_text('not an archive')
        logs = self.root / 'logs'
        cmd = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'tools/extract-asura-batch.ps1'), '-GameRoot', str(game), '-ExtractRoot', str(self.root / 'out'), '-LogsRoot', str(logs), '-PythonExe', sys.executable, '-NativeExtractor', str(ROOT / 'tools/extract-asura-native.py')]
        unified = [sys.executable, str(ROOT / 'tools/bzg-extractor.py'), 'extract', '--game-root', str(game), '--extract-root', str(self.root / 'out'), '--logs-root', str(logs), '--no-skip-existing']
        result = subprocess.run(unified, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with (logs / 'extract-report.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(r['Status'] == 'OK' for r in rows), 5)
        self.assertEqual(sum(r['Status'] == 'Pointer' for r in rows), 1)
        # Unsupported inputs must make the pipeline fail rather than claim success.
        (game / 'unsupported.pc').write_bytes(b'AsuraCmp' + bytes(32))
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
