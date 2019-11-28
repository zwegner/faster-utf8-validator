import sys

error_nibbles = [
#    [0, [[0x8, 0xB], [0x0,  -1], [0x0,  -1]]],
#    [1, [[0xC, 0xD], [0x0,  -1], [0x0,  -1]]],
#    [1, [[0xE, 0xE], [0x0,  -1], [0x0,  -1]]],
#    [3, [[0xE, 0xE], [0x0,  -1], [0x0,  -1]]],
#    [1, [[0xF, 0xF], [0x0,  -1], [0x0,  -1]]],
#    [3, [[0xF, 0xF], [0x0,  -1], [0x0,  -1]]],
#    [5, [[0xF, 0xF], [0x0,  -1], [0x0,  -1]]],
#
#    [2, [ 0xC,       [0x0, 0x1], [0x0, 0xF]]],
#    [4, [ 0xE,        0x0,       [0x8, 0x9]]],
#    [3, [ 0xE,        0xD,       [0xA, 0xB]]],
#    [5, [ 0xF,        0x0,       [0x0, 0x8]]],
#    [6, [ 0xF,        0x4,       [0x9, 0xF]]],
#    [7, [ 0xF,       [0x5, 0xF], [0x0, 0xF]]],

    [0, [ 0xC,       [0x0, 0x1], [0x0, 0xF]]],
    [1, [ 0xE,        0x0,       [0x8, 0x9]]],
    [2, [ 0xE,        0xD,       [0xA, 0xB]]],
    [3, [ 0xF,        0x0,       [0x0, 0x8]]],
    [4, [ 0xF,        0x4,       [0x9, 0xF]]],
    #[5, [ 0xF,       [0x5, 0x7], [0x0, 0xF]]],
    #[6, [ 0xF,       [0x8, 0xF], [0x0, 0xF]]],
    [5, [ 0xF,       [0x5, 0xF], [0x0, 0xF]]],
]
error_nibbles = [[bit, [[n, n] if isinstance(n, int) else n for n in nibbles]]
        for bit, nibbles in error_nibbles]

error_bits = [[0] * 16 for i in range(3)]
error_bits_64 = [[0] * 64 for i in range(3)]
error_bits_64_2 = [[0] * 64 for i in range(3)]
error_bits_64_3 = [[0] * 64 for i in range(3)]

for bit, nibbles in error_nibbles:
    for n, (lo, hi) in enumerate(nibbles):
        for value in range(lo, hi+1):
            error_bits[n][value] |= 1 << bit

for n in range(64):
    n_1 = n_2 = n_3 = 0
    for x in range(4):
        n_1 |= error_bits[0][n >> 2] & error_bits[1][((n & 3) << 2) | x]
    for x in range(4):
        n_2 |= error_bits[2][n & 15] & error_bits[1][(n >> 4) | (x << 2)]
        #n_3 |= error_bits[2][n & 15] & error_bits[1][(n >> 4) | 8 | (x << 2)]
    error_bits_64[0][n] = n_1
    error_bits_64[1][n] = n_2
    error_bits_64[2][n] = n_3

for n in range(64):
    n_1 = n_2 = n_3 = 0
    for x in range(4):
        n_1 |= error_bits[0][(n >> 4) | (x << 2)] & error_bits[1][n & 15]
    for x in range(4):
        n_2 |= error_bits[2][n & 15] & error_bits[0][(n & 0x30) >> 2 | x]
    error_bits_64_3[0][n] = n_1
    error_bits_64_3[1][n] = n_2
    error_bits_64_3[2][n] = n_3

for n in range(64):
    n_1 = n_2 = n_3 = 0
    for x in range(4):
        n_1 |= error_bits[0][8 | n >> 4] & error_bits[1][n & 15]
        n_2 |= error_bits[0][12 | n >> 4] & error_bits[1][n & 15]
    for x in range(4):
        n_3 |= error_bits[2][n & 15]
    error_bits_64_2[0][n] = n_1
    error_bits_64_2[1][n] = n_2
    error_bits_64_2[2][n] = n_3

with open(sys.argv[1], 'w') as f:
    f.write('#if PERM1\n')
    for n in range(2):
        table = ',\n    '.join(', '.join('0x%02X' % v for v in error_bits_64[n][x:x+8])
                for x in range(0, 64, 8))
        f.write('const vec_t error_64_%s = V_TABLE_64(\n    %s\n);\n' % (n+1, table))
    f.write('#elif PERM2\n')
    for n in range(3):
        table = ',\n    '.join(', '.join('0x%02X' % v for v in error_bits_64_2[n][x:x+8])
                for x in range(0, 64, 8))
        f.write('const vec_t error_64_%s = V_TABLE_64(\n    %s\n);\n' % (n+1, table))
    f.write('#elif PERM3\n')
    for n in range(2):
        table = ',\n    '.join(', '.join('0x%02X' % v for v in error_bits_64_3[n][x:x+8])
                for x in range(0, 64, 8))
        f.write('const vec_t error_64_%s = V_TABLE_64(\n    %s\n);\n' % (n+1, table))
    f.write('#else\n')
    for n in range(3):
        table = ',\n    '.join(', '.join('0x%02X' % v for v in error_bits[n][x:x+4])
                for x in range(0, 16, 4))
        f.write('const vec_t error_%s = V_TABLE_16(\n    %s\n);\n' % (n+1, table))
    f.write('#endif\n')

#t = 0
#for a in range(2):
#    for b in range(2):
#        for c in range(2):
#            x = c if a else b
#            t |= x << (a<<2 | b<<1 | c)
#print(hex(t))
