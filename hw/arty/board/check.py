#!/usr/bin/env python3
"""Run bounded learned-policy inference comparisons over Linux USB UART.

The supplied fixture packets contain unmodified trained FP32 parameters and
prepared observations. Per-job weights are streamed; no DDR/cache or periodic
control-rate claim is made. No automatic retry, reset, flush, or skipped failure.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import struct
import sys
import termios
import time
import zlib


def decode_reply(packet, job, tolerance=1e-5):
    if len(packet) != 32 or packet[:4] != b'VRO1':
        raise ValueError(f'Invalid reply frame: {packet.hex()}')
    if zlib.crc32(packet[:28]) != struct.unpack_from('<I', packet, 28)[0]:
        raise ValueError('Reply CRC32 mismatch')
    version, actor, fault, flags, number, x, y, cycles, request_crc = struct.unpack_from('<BBBBIffII', packet, 4)
    if (version, actor, number, request_crc) != (1, job['actor'], job['job'], job['request_crc32']):
        raise ValueError(f'Reply identity mismatch: {(version, actor, number, request_crc)}')
    if fault:
        raise ValueError(f'FPGA sticky fault 0x{fault:02x}; explicit reprogram/reset required')
    if flags & 0x1c:
        raise ValueError(f'Invalid/divide-by-zero/overflow arithmetic flags: 0x{flags:02x}')
    if not all(math.isfinite(v) and abs(v) <= 1 for v in [x, y]):
        raise ValueError(f'Invalid tanh outputs: {[x, y]}')
    error = max(abs(v - e) for v, e in zip([x, y], job['expected']))
    result = {'job': number, 'actor': ('inner', 'outer')[actor],
              'golden_index': job['golden_index'], 'actual': [x, y],
              'expected': job['expected'], 'max_abs_error': error,
              'fault': fault, 'arithmetic_flags': flags,
              'cycles_including_uart_starvation': cycles, 'atol': tolerance,
              'passed': error <= tolerance}
    if not result['passed']:
        raise ValueError(f'Inference mismatch: {json.dumps(result)}')
    return result


def write_all(fd, data, timeout=30):
    deadline = time.monotonic() + timeout
    offset = 0
    while offset < len(data):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
            raise TimeoutError(f'TX stalled after {offset}/{len(data)} bytes')
        try:
            offset += os.write(fd, data[offset:offset + 16384])
        except BlockingIOError:
            continue
    return offset


def read_reply(fd, raw, timeout=10):
    deadline = time.monotonic() + timeout
    data = bytearray()
    while len(data) < 32:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            raise TimeoutError(f'Incomplete FPGA reply ({len(data)}/32 bytes): {data.hex()}')
        part = os.read(fd, 32 - len(data))
        if not part:
            raise RuntimeError('Serial device disconnected')
        raw.write(part)
        raw.flush()
        data += part
    return bytes(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--port', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error('--limit must be positive')
    a.out.mkdir(parents=True, exist_ok=False)
    fixtures = json.loads((a.fixtures / 'fixtures.json').read_text())
    report = {'status': 'running', 'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'checkpoint_sha256': fixtures['checkpoint_sha256'],
              'golden_vectors_sha256': fixtures['vectors_sha256'],
              'port': str(a.port), 'python': sys.version, 'baud': 1000000, 'format': '8N1',
              'scope': 'Physical FP32 inference parity; streamed weights; no periodic control-rate claim',
              'atol': fixtures['atol'], 'rtol': 0, 'jobs': [], 'planned_jobs': 0}
    fd = None
    original = None
    started = time.monotonic()
    try:
        jobs = fixtures['board_jobs'][:a.limit] if a.limit else fixtures['board_jobs']
        report['planned_jobs'] = len(jobs)
        fd = os.open(a.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        # pyserial's exclusive mode uses flock. TIOCEXCL alone cannot exclude
        # readers that opened the device before us; acquire both conventions.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.ioctl(fd, termios.TIOCEXCL)
        original = termios.tcgetattr(fd)
        settings = termios.tcgetattr(fd)
        settings[0] = 0
        settings[1] = 0
        settings[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
        settings[3] = 0
        settings[4] = settings[5] = termios.B1000000
        settings[6][termios.VMIN] = 0
        settings[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, settings)
        with (a.out / 'trace.jsonl').open('x') as trace:
            for job in jobs:
                record = {'job': job['job'], 'actor': job['actor'], 'golden_index': job['golden_index'],
                          'status': 'started', 'request_file': str(a.fixtures / job['file'])}
                report['active_job'] = record
                request = (a.fixtures / job['file']).read_bytes()
                assert len(request) == job['bytes']
                assert zlib.crc32(request[4:-4]) == job['request_crc32'] == struct.unpack_from('<I', request, len(request) - 4)[0]
                record['request_sha256'] = hashlib.sha256(request).hexdigest()
                begin = time.monotonic()
                with (a.out / f"reply-{job['job']:03d}.bin").open('xb') as raw:
                    if select.select([fd], [], [], 0)[0]:
                        unexpected = os.read(fd, 4096)
                        if unexpected:
                            raw.write(unexpected)
                            raise ValueError(f'Unexpected buffered input: {unexpected.hex()}')
                    record['sent_bytes'] = write_all(fd, request)
                    packet = read_reply(fd, raw)
                result = decode_reply(packet, job, fixtures['atol'])
                result.update(wall_seconds=time.monotonic() - begin, request_sha256=record['request_sha256'])
                report['jobs'].append(result)
                report.pop('active_job', None)
                trace.write(json.dumps(result) + '\n')
                trace.flush()
                print(f"PASS {result['actor']} vector={job['golden_index']} max_abs_error={result['max_abs_error']:.9g}", flush=True)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        print(report['error'], flush=True)
    finally:
        if fd is not None:
            try:
                if original is not None:
                    termios.tcsetattr(fd, termios.TCSANOW, original)
            except OSError as error:
                report.update(status='failed', close_error=str(error))
            finally:
                os.close(fd)
        report['elapsed_s'] = time.monotonic() - started
        report['max_abs_error'] = max((j['max_abs_error'] for j in report['jobs']), default=None)
        (a.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'jobs': len(report['jobs']),
                          'max_abs_error': report['max_abs_error'], 'elapsed_s': report['elapsed_s']}, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
