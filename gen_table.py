import sys

error_nibbles = [
    # First continuation byte errors
    [0, [[0xC, 0xF], [0x0, 0xF], [0x0, 0x7]]],
    [0, [[0xC, 0xF], [0x0, 0xF], [0xC, 0xF]]],

    [1, [ 0xC,       [0x0, 0x1], [0x0, 0xF]]],
    [2, [ 0xE,        0x0,       [0x8, 0x9]]],
    [3, [ 0xE,        0xD,       [0xA, 0xB]]],
    [4, [ 0xF,        0x0,       [0x0, 0x8]]],
    [5, [ 0xF,        0x4,       [0x9, 0xF]]],
    [6, [ 0xF,       [0x5, 0xF], [0x0, 0xF]]],

    # Continuation bytes. The second nibble does not have any values here,
    # to make sure the final error mask will not be triggered.
    [7, [[0x8, 0xB], [0x0,  -1], [0x8, 0xB]]],
]

# Transform single values to [n, n] ranges
error_nibbles = [[bit, [[n, n] if isinstance(n, int) else n for n in nibbles]]
        for bit, nibbles in error_nibbles]

# Compute the three 16-entry lookup tables based on the three nibbles
error_bits = [[0] * 16 for i in range(3)]
for bit, nibbles in error_nibbles:
    for n, (lo, hi) in enumerate(nibbles):
        for value in range(lo, hi+1):
            error_bits[n][value] |= 1 << bit

# Compute the two 64-entry lookup tables for AVX-512 by merging the three
# 16-entry tables
error_bits_64 = [[0] * 64 for i in range(2)]
for n in range(64):
    n_1 = n_2 = 0
    for x in range(4):
        n_1 |= error_bits[0][(n >> 4) | (x << 2)] & error_bits[1][n & 15]
        n_2 |= error_bits[2][n & 15] & error_bits[0][(n & 0x30) >> 2 | x]
    error_bits_64[0][n] = n_1
    error_bits_64[1][n] = n_2

# Write the output file
with open(sys.argv[1], 'w') as f:
    f.write('#if defined(AVX512_VBMI)\n')
    for n in range(2):
        table = ',\n    '.join(', '.join('0x%02X' % v for v in error_bits_64[n][x:x+8])
                for x in range(0, 64, 8))
        f.write('const vec_t error_64_%s = V_TABLE_64(\n    %s\n);\n' % (n+1, table))
    f.write('#else\n')
    for n in range(3):
        table = ',\n    '.join(', '.join('0x%02X' % v for v in error_bits[n][x:x+4])
                for x in range(0, 16, 4))
        f.write('const vec_t error_%s = V_TABLE_16(\n    %s\n);\n' % (n+1, table))
    f.write('#endif\n')
