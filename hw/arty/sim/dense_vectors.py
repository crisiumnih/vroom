"""Synthetic FP32 fixtures for the dense microkernel, not checkpoint exports.

Reference is bias-first sequential dot product, with a binary32 rounding after
each multiply and each add. Python binary64 holds each binary32 product exactly;
struct converts results to binary32. Inputs stay finite and far from overflow.
The C VPI operator uses native float, providing an independent execution path.
"""
import hashlib
import json
from pathlib import Path
import random
import struct


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def generate(output: Path, seed: int):
    rng = random.Random(seed)
    cases = [("inner_hidden", 25, 256), ("outer_hidden", 19, 256),
             ("inner_output", 256, 2), ("inner_middle", 256, 256)]
    activations, biases, weights, expected, small_expected = [], [], [], [], []
    alignment_expected = []
    for name, ninput, noutput in cases:
        x = [f32(rng.uniform(-0.9, 0.9)) for _ in range(ninput)]
        bias = [f32(rng.uniform(-0.1, 0.1)) for _ in range(noutput)]
        matrix = [[f32(rng.uniform(-0.25, 0.25)) for _ in range(ninput)] for _ in range(noutput)]
        y = []
        for row, acc in zip(matrix, bias):
            for w, value in zip(row, x):
                acc = f32(acc + f32(w * value))
            y.append(acc)
        if name == "inner_output":
            for row, acc in zip(matrix, bias):
                for w, value in zip(row[:4], x[:4]):
                    acc = f32(acc + f32(w * value))
                small_expected.append(bits(acc))
        if name == "inner_middle":
            for row, acc in zip(matrix, bias):
                for w, value in zip(row[:4], x[:4]):
                    acc = f32(acc + f32(w * value))
                alignment_expected.append(bits(acc))
        activations += [bits(v) for v in x] + [0] * (256-ninput)
        biases += [bits(v) for v in bias] + [0] * (256-noutput)
        expected += [bits(v) for v in y] + [0] * (256-noutput)
        for row in matrix:
            weights += [bits(v) for v in row] + [0] * (256-ninput)
        weights += [0] * ((256-noutput)*256)
    files = {}
    for name, values in [("inputs", activations), ("bias", biases), ("weights", weights),
                         ("expected", expected), ("small-expected", small_expected),
                         ("alignment-expected", alignment_expected)]:
        path = output / f"dense-{name}.hex"
        path.write_text("".join(f"{value:08x}\n" for value in values))
        files[path.name] = {"words": len(values), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    metadata = {"source": "synthetic; no checkpoint data", "seed": seed, "cases": cases,
                "order": "bias first; ascending input columns; separate binary32 multiply/add rounding", "files": files}
    (output / "dense-fixtures.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata
