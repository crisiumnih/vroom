#!/usr/bin/env python3
"""Prepare existing FP32 checkpoint/golden vectors for stream and board checks.

No training or quantization. Matrices retain [output,input] binary32 bits.
The RTL dot order is bias-first, separately rounded multiply then add, ascending
input column; this is compared separately from the package BLAS reference.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import zlib

import numpy as np


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial_packet(weights, actor, observation, job, lanes):
    """VRM1, actor:u8, job:u32, input words, each layer bias/weights, CRC32.

    FP32 words/u32 are little endian. Weights: column, group, lane. Biases:
    group, lane. Inactive tail lanes carry explicit transport zeros and are
    masked by RTL; these are not padded observations or missing trace samples.
    CRC covers actor through final weight, using the standard zlib convention.
    """
    payload = bytearray(struct.pack('<BI', actor, job))
    payload += observation.astype('<f4').tobytes()
    name = ('inner', 'outer')[actor]
    for layer in range(3):
        w, b = weights[f'{name}_W{layer}'], weights[f'{name}_b{layer}']
        rows, cols = w.shape
        padded_rows = ((rows + lanes - 1) // lanes) * lanes
        biases = np.zeros(padded_rows, dtype='<f4')
        matrix = np.zeros((cols, padded_rows), dtype='<f4')
        biases[:rows] = b
        matrix[:, :rows] = w.T
        payload += biases.tobytes() + matrix.tobytes()
    crc = zlib.crc32(payload)
    return b'VRM1' + payload + struct.pack('<I', crc), crc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lanes', type=int, default=12, choices=[12, 16])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    arty = Path(__file__).resolve().parents[1]
    fp = module('board_fp_reference', arty / 'fp32/sim/reference.py')
    tanh = module('board_tanh_reference', arty / 'tanh/sim/model.py')
    ref = module('package_reference', args.package / 'inference_reference.py')
    checkpoint = args.package / 'selected_checkpoint.npz'
    vectors = args.package / 'golden_vectors.npz'
    hashes = json.loads((args.package / 'actor_export_manifest.json').read_text())
    assert sha(checkpoint) == hashes[checkpoint.name]
    assert sha(vectors) == hashes[vectors.name]
    actors = ref.Actors(checkpoint)
    software = ref.verify_vectors(checkpoint, vectors)
    (args.out / 'software_parity.json').write_text(json.dumps(software, indent=2) + '\n')
    assert software['passed'], 'Package reference/golden mismatch'
    report = {'status': 'running', 'checkpoint_sha256': sha(checkpoint),
              'vectors_sha256': sha(vectors), 'numpy': np.__version__,
              'lanes': args.lanes, 'rtol': 0, 'atol': 1e-5,
              'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=arty, text=True).strip(),
              'jobs': [], 'board_jobs': []}
    def hexfile(name, values):
        (args.out / name).write_text(''.join(f'{int(v):08x}\n' for v in values))
    with np.load(vectors, allow_pickle=False) as gold:
        for job, (actor, index) in enumerate([(0, 0), (1, 0), (0, 517), (1, 517)]):
            name = ('inner', 'outer')[actor]
            x = gold[f'{name}_input'][index]
            bank = [int(v) for v in x.view(np.uint32)]
            all_pre, all_post, all_weights, all_bias = [], [], [], []
            flags = 0
            for layer in range(3):
                w = actors.weights[f'{name}_W{layer}'].view(np.uint32)
                b = actors.weights[f'{name}_b{layer}'].view(np.uint32)
                pre, post = [], []
                for row in range(len(b)):
                    total = int(b[row])
                    for col in range(w.shape[1]):
                        product, mf = fp.operate(int(w[row, col]), bank[col], True)
                        total, af = fp.operate(total, product)
                        flags |= mf | af
                    pre.append(total)
                    if layer < 2:
                        value = 0 if total >> 31 and total & 0x7fffffff else total
                    else:
                        value, invalid, _, _ = tanh.evaluate(total)
                        assert not invalid
                    post.append(value)
                all_pre += pre
                all_post += post
                all_weights += [int(v) for v in w.T.ravel()]
                all_bias += [int(v) for v in b]
                bank = post
            actual = np.array(bank, dtype=np.uint32).view(np.float32)
            expected = gold[f'{name}_output'][index]
            error = float(np.max(np.abs(actual - expected)))
            assert error <= 1e-5, (name, index, error)
            for suffix, data in [('input', x.view(np.uint32)), ('weight', all_weights),
                                 ('bias', all_bias), ('pre', all_pre), ('post', all_post), ('output', bank)]:
                hexfile(f'job{job}-{suffix}.hex', data)
            (args.out / f'job{job}-flags.hex').write_text(f'{flags:02x}\n')
            report['jobs'].append({'job': job, 'actor': actor, 'golden_index': index,
                                   'output_hex': [f'{v:08x}' for v in bank],
                                   'max_abs_error_vs_package': error, 'flags': flags})
        indices = [0, 1, 2, 17, 81, 255, 400, 512, 513, 514, 515, 516, 517]
        for index in indices:
            for actor, name in enumerate(['inner', 'outer']):
                job = len(report['board_jobs']) + 1
                packet, crc = serial_packet(actors.weights, actor, gold[f'{name}_input'][index], job, args.lanes)
                filename = f'board-job{job:03d}.bin'
                (args.out / filename).write_bytes(packet)
                report['board_jobs'].append({'job': job, 'actor': actor, 'golden_index': index,
                    'file': filename, 'bytes': len(packet), 'request_crc32': crc,
                    'expected': gold[f'{name}_output'][index].tolist()})
    report['status'] = 'passed'
    (args.out / 'fixtures.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'software_vectors_per_actor': 518,
                      'rtl_jobs': len(report['jobs']), 'board_jobs': len(report['board_jobs']),
                      'out': str(args.out)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
