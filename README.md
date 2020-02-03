# faster-utf8-validator
This library is a very fast UTF-8 validator using SIMD instructions. It
supports SSE4, AVX2, and AVX-512 on x86, as well as ARM NEON. As
far as I am aware, it is the fastest validator in the world on the CPUs that
support these instructions. Using AVX2, it can validate random UTF-8 text as
fast as .19 cycles/byte, and random ASCII text at .04
cycles/byte. For UTF-8, this is roughly 2x faster than the
[fastvalidate-utf-8](https://github.com/lemire/fastvalidate-utf-8) library.

This repository contains the library (one C file, `z_validate.c`), a lookup
table generation script (`gen_table.py`), a build script for the
[make.py](https://github.com/zwegner/make.py) build system, and a Lua test
script (which requires LuaJIT due to use of the `ffi` module).

A detailed description of the algorithm can be found in `z_validate.c`, with
many comments about the table layout in `gen_table.py` (essential for fully
understanding the algorithm).

Now, even faster and more architecure-supporting-y
---

As of 2020-02-02, besides adding support for AVX-512 and NEON, this library has
been sped up even further. Depending on the benchmarking method, it can be up to
roughly 40% faster than the original published version.

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


| Validator                             | 64K UTF-8 | 64K ASCII | 16M UTF-8 | 16M ASCII |
| ------------------------------------- | --------- | --------- | --------- | --------- |
| `validate_utf8_fast_avx`              | 0.3477    | 0.3473    | 0.3559    | 0.3586    |
| `validate_utf8_fast_avx_asciipath`    | 0.3420    | 0.0623    | 0.3849    | 0.1403    |
| `validate_utf8_fast_avx512`           | 0.2377    | 0.2370    | 0.2677    | 0.2675    |
| `validate_utf8_fast_avx512_asciipath` | 0.2453    | 0.0321    | 0.2731    | 0.1177    |
| `z_validate_utf8_avx512_vbmi`         | 0.0769    | 0.0772    | 0.1425    | 0.1430    |
| `z_validate_utf8_avx512_vbmi_ascii`   | 0.0831    | 0.0287    | 0.1406    | 0.1166    |
| `z_validate_utf8_avx2`                | 0.2157    | 0.2157    | 0.2412    | 0.2412    |
| `z_validate_utf8_sse4`                | 0.4576    | 0.4577    | 0.4625    | 0.4629    |


| Validator                             | 64K UTF-8 | 64K ASCII | 16M UTF-8 | 16M ASCII |
| ------------------------------------- | --------- | --------- | --------- | --------- |
| `validate_utf8_fast_avx`              |    0.3469 |    0.3476 |    0.3582 |    0.3672 |
| `validate_utf8_fast_avx_asciipath`    |    0.3419 |    0.0613 |    0.3849 |    0.1397 |
| `validate_utf8_fast_avx512`           |    0.2377 |    0.2370 |    0.2674 |    0.2659 |
| `validate_utf8_fast_avx512_asciipath` |    0.2452 |    0.0320 |    0.2728 |    0.1176 |
| `z_validate_utf8_avx512_vbmi`         |    0.0771 |    0.0772 |    0.1436 |    0.1405 |
| `z_validate_utf8_avx512_vbmi_ascii`   |    0.0870 |    0.0275 |    0.1464 |    0.1164 |
| `z_validate_utf8_avx2`                |    0.1587 |    0.1590 |    0.2025 |    0.2023 |
| `z_validate_utf8_avx2_ascii`          |    0.1706 |    0.0356 |    0.2162 |    0.1208 |
| `z_validate_utf8_sse4`                |    0.3345 |    0.3348 |    0.3432 |    0.3434 |
| `z_validate_utf8_sse4_ascii`          |    0.3496 |    0.0672 |    0.3582 |    0.1554 |


| Validator                             | 64K UTF-8 | 64K ASCII | 16M UTF-8 | 16M ASCII |
| ------------------------------------- | --------- | --------- | --------- | --------- |
| `validate_utf8_fast_avx`              |    0.3475 |    0.3469 |    0.3587 |    0.3584 |
| `validate_utf8_fast_avx_asciipath`    |    0.3413 |    0.0613 |    0.3845 |    0.1400 |
| `validate_utf8_fast_avx512`           |    0.2376 |    0.2368 |    0.2678 |    0.2728 |
| `validate_utf8_fast_avx512_asciipath` |    0.2452 |    0.0318 |    0.2726 |    0.1184 |
| `z_validate_utf8_avx512_vbmi`         |    0.0768 |    0.0767 |    0.1425 |    0.1418 |
| `z_validate_utf8_avx512_vbmi_ascii`   |    0.0845 |    0.0217 |    0.1439 |    0.1050 |
| `z_validate_utf8_avx2`                |    0.1531 |    0.1534 |    0.1980 |    0.1978 |
| `z_validate_utf8_avx2_ascii`          |    0.1726 |    0.0304 |    0.2066 |    0.1158 |
| `z_validate_utf8_sse4`                |    0.3382 |    0.3268 |    0.3377 |    0.3369 |
| `z_validate_utf8_sse4_ascii`          |    0.3673 |    0.0517 |    0.3623 |    0.1229 |


