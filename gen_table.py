import sys

error_nibbles = [
    # First continuation byte errors
    ['ERR_CONT',  [[0xC, 0xF], [0x0, 0xF], [0x0, 0x7]]],
    ['ERR_CONT',  [[0xC, 0xF], [0x0, 0xF], [0xC, 0xF]]],

    ['ERR_OVER1', [ 0xC,       [0x0, 0x1], [0x0, 0xF]]],
    ['ERR_OVER2', [ 0xE,        0x0,       [0x8, 0x9]]],
    ['ERR_SURR',  [ 0xE,        0xD,       [0xA, 0xB]]],
    ['ERR_OVER3', [ 0xF,        0x0,       [0x0, 0x8]]],
    ['ERR_MAX1',  [ 0xF,        0x4,       [0x9, 0xF]]],
    ['ERR_MAX2',  [ 0xF,       [0x5, 0xF], [0x0, 0xF]]],

    # Phony bit that overlaps with surrogate pair errors
    ['ERR_SURR',  [ 0xF,       [0x0,  -1], [0x0,  -1]]],

    # Continuation bytes. The second nibble does not have any values here,
    # to make sure the final error mask will not be triggered.
    ['MARK_CONT', [[0x8, 0xB], [0x0,  -1], [0x8, 0xB]]],
]

x86_bit_map = {
    'ERR_CONT':   0,
    'ERR_OVER1':  1,
    'ERR_OVER2':  2,
    'ERR_SURR':   3,
    'ERR_OVER3':  4,
    'ERR_MAX1':   5,
    'ERR_MAX2':   6,
    'MARK_CONT':  7
}
neon_bit_map = {
    'ERR_CONT':   1,
    'ERR_OVER1':  2,
    'ERR_OVER2':  3,
    'ERR_SURR':   4,
    'ERR_OVER3':  5,
    'ERR_MAX1':   6,
    'ERR_MAX2':   7,
    'MARK_CONT':  0
}

def make_bit_tables(bit_map):
    # Transform single values to [n, n] ranges, and remap bits
    errors = [[bit_map[bit], [[n, n] if isinstance(n, int) else n
        for n in nibbles]] for bit, nibbles in error_nibbles]

    # Compute the three 16-entry lookup tables based on the three nibbles
    error_bits = [[0] * 16 for i in range(3)]
    for bit, nibbles in errors:
        for n, (lo, hi) in enumerate(nibbles):
            for value in range(lo, hi+1):
                error_bits[n][value] |= 1 << bit

    return error_bits

# Compute the two 64-entry lookup tables for AVX-512 by merging the three
# 16-entry tables
def make_64_bit_tables(error_bits):
    error_bits_64 = [[0] * 64 for i in range(2)]
    for n in range(64):
        n_1 = n_2 = 0
        for x in range(4):
            n_1 |= error_bits[0][(n >> 4) | (x << 2)] & error_bits[1][n & 15]
            n_2 |= error_bits[2][n & 15] & error_bits[0][(n & 0x30) >> 2 | x]
        error_bits_64[0][n] = n_1
        error_bits_64[1][n] = n_2
    return error_bits_64

def write_table(f, table, bit_map):
    t_len = len(table[0])
    step = 8 if t_len == 64 else 4
    for n in range(len(table)):
        t = ',\n    '.join(', '.join('0x%02X' % v for v in table[n][x:x+step])
                for x in range(0, t_len, step))
        f.write('const vec_t error_%s = V_TABLE_%s(\n    %s\n);\n' % (n+1, t_len, t))
    #f.write('enum {\n')
    #for k, v in sorted(bit_map.items()):
    #    f.write('    %s = %s,\n' % (k, v))
    #f.write('};\n')
    for k, v in sorted(bit_map.items()):
        f.write('#   define %-20s %s\n' % (k, v))

def main(path):
    x86_table = make_bit_tables(x86_bit_map)
    avx512_table = make_64_bit_tables(x86_table)
    neon_table = make_bit_tables(neon_bit_map)

    # Write the output file
    with open(path, 'w') as f:
        f.write('#if defined(AVX512_VBMI)\n')
        write_table(f, avx512_table, x86_bit_map)
        f.write('#elif defined(NEON)\n')
        write_table(f, neon_table, neon_bit_map)
        f.write('#else\n')
        write_table(f, x86_table, x86_bit_map)
        f.write('#endif\n')

if __name__ == '__main__':
    main(sys.argv[1])
