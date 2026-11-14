# Pinned Berkeley HardFloat subset

Unmodified Verilog from John Hauser's **HardFloat Release 1 (2019-07-29)**,
downloaded from https://www.jhauser.us/arithmetic/HardFloat-1.zip.
`upstream.json` pins the archive and each vendored file by SHA-256. The archive
hash is `6b3757c9fbfa2230c6a2b84605e39372cb589dd7500e979c4f0b8ecc8a03b14b`.

Only addition, multiplication, format conversion and their common dependencies
are retained, along with the original license and package README. The RISCV
specialization produces canonical quiet NaNs. No upstream file is edited.
Repository-owned pipeline/accumulator wrappers live in `hw/arty/fp32`.

See `COPYING.txt` and the source headers for the three-clause BSD license.
Official documentation: https://www.jhauser.us/arithmetic/HardFloat-1/doc/HardFloat-Verilog.html.
