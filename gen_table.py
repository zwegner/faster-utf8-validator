import sys

error_nibbles = [
    [0, [[0x8, 0xB], [0x0,  -1], [0x0,  -1]]],
    [1, [[0xC, 0xD], [0x0,  -1], [0x0,  -1]]],
    [1, [[0xE, 0xE], [0x0,  -1], [0x0,  -1]]],
    [3, [[0xE, 0xE], [0x0,  -1], [0x0,  -1]]],
    [1, [[0xF, 0xF], [0x0,  -1], [0x0,  -1]]],
    [3, [[0xF, 0xF], [0x0,  -1], [0x0,  -1]]],
    [5, [[0xF, 0xF], [0x0,  -1], [0x0,  -1]]],

    [2, [ 0xC,       [0x0, 0x1], [0x0, 0xF]]],
    [4, [ 0xE,        0x0,       [0x8, 0x9]]],
    [3, [ 0xE,        0xD,       [0xA, 0xB]]],
    [5, [ 0xF,        0x0,       [0x0, 0x8]]],
    [6, [ 0xF,        0x4,       [0x9, 0xF]]],
    [7, [ 0xF,       [0x5, 0xF], [0x0, 0xF]]],
]
error_nibbles = [[bit, [[n, n] if isinstance(n, int) else n for n in nibbles]]
        for bit, nibbles in error_nibbles]

error_bits = [[0] * 16 for i in range(3)]

for bit, nibbles in error_nibbles:
    for n, (lo, hi) in enumerate(nibbles):
        for value in range(lo, hi+1):
            error_bits[n][value] |= 1 << bit

with open(sys.argv[1], 'w') as f:
    for n in range(3):
        table = ',\n    '.join(', '.join('0x%02X' % v for v in error_bits[n][x:x+4])
                for x in range(0, 16, 4))
        f.write('static vec_t error_%s = V_TABLE_16(\n    %s\n);\n' % (n+1, table))
