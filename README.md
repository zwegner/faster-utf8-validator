# faster-utf8-validator
This library is a very fast UTF-8 validator using AVX2/SSE4 instructions. As
far as I am aware, it is the fastest validator in the world on the CPUs that
support these instructions (...and not AVX-512). Using AVX2, it can validate
random UTF-8 text as fast as .26 cycles/byte, and random ASCII text at .09
cycles/byte. For UTF-8, this is roughly 1.5-1.7x faster than the
[fastvalidate-utf-8](https://github.com/lemire/fastvalidate-utf-8) library.

This repository contains the library (one C file), a build script for the
[make.py](https://github.com/zwegner/make.py) build system, and a Lua test
script (which requires LuaJIT due to use of the `ffi` module).

A detailed description of the algorithm can be found in `z_validate.c`.
This algorithm should map fairly nicely to AVX-512, and should in fact be a
bit faster than 2x the speed of AVX2 since a few instructions can be saved.
But I don't have an AVX-512 machine, so I haven't tried it yet.

Benchmark
----
Here's some raw numbers, measured on my 2.4GHz Haswell laptop, using a modified
version of the benchmark in the fastvalidate-utf-8 repository. There are four
configurations of test input: random UTF-8 bytes or random ASCII bytes, and
either 64K bytes or 16M bytes. All measurements are the best of 50 runs, with
each run using a different random seed, but each validator tested with the
same seeds (and thus the same inputs). All measurements are in cycles per byte.
The first two rows are the fastvalidate-utf-8 AVX2 functions, and the second two
rows are this library, using AVX2 and SSE4 instruction sets.

| Validator                          | 64K UTF-8 | 64K ASCII | 16M UTF-8 | 16M ASCII |
| ---------------------------------- | --------- | --------- | --------- | --------- |
| `validate_utf8_fast_avx`           |     0.410 |     0.410 |     0.496 |     0.429 |
| `validate_utf8_fast_avx_asciipath` |     0.436 |     0.074 |     0.457 |     0.156 |
| `z_validate_utf8_avx2`             |     0.264 |     0.079 |     0.290 |     0.160 |
| `z_validate_utf8_sse4`             |     0.568 |     0.163 |     0.596 |     0.202 |
