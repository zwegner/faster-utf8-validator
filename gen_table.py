import sys

error_nibbles = [
    # First continuation byte errors. Any byte following 0xCx..0xFx must be
    # in the range 0x80..0xBF
    ['ERR_CONT',  [[0xC, 0xF], [0x0, 0xF], [0x0, 0x7]]],
    ['ERR_CONT',  [[0xC, 0xF], [0x0, 0xF], [0xC, 0xF]]],

    ['ERR_OVER1', [ 0xC,       [0x0, 0x1], [0x0, 0xF]]],
    ['ERR_OVER2', [ 0xE,        0x0,       [0x8, 0x9]]],
    ['ERR_SURR',  [ 0xE,        0xD,       [0xA, 0xB]]],
    ['ERR_OVER3', [ 0xF,        0x0,       [0x0, 0x8]]],
    ['ERR_MAX1',  [ 0xF,        0x4,       [0x9, 0xF]]],
    ['ERR_MAX2',  [ 0xF,       [0x5, 0xF], [0x0, 0xF]]],

    # Phony bit that overlaps with surrogate pair errors. This means we have a
    # single bit in the error mask that is set for 0xEx and 0xFx, which allows
    # us to find both 3 and 4 byte sequences with just one test.
    ['ERR_SURR',  [ 0xF,       [0x0,  -1], [0x0,  -1]]],

    # Continuation bytes. The second nibble does not have any values here, to
    # make sure the final error mask will never be triggered--this is only a bit
    # that we pull out for a separate continuation byte test. We also use the
    # fact that we test contiguous bytes for the bit mask here, to only get
    # *trailing* continuation bytes: the first and third nibbles here both must
    # be in the continuation byte range. This means we need to AND the first and
    # third nibble's masks together first to pull this bit out, before ANDing
    # with the second nibble. Since the second nibble will never have this bit
    # set, it will never cause an error in the special case testing. We're
    # already ANDing all three nibbles together, though, so we can use the
    # intermediate result of the first and third together for free. All of this
    # together means we only need to test the second and third continuation
    # bytes aside from the special case testing, since the first continuation
    # byte is handled with the ERR_CONT bit in this table.
    # The above explanation is slightly different in the case of AVX-512. There,
    # we only have two lookups, so we can't use the intermediate result from the
    # first and third nibble. Luckily, as described below, we organize the
    # tables to get the continuation bit for free on AVX-512 as well.
    ['MARK_CONT', [[0x8, 0xB], [0x0,  -1], [0x8, 0xB]]],
]

# Bit maps: the different error bits are arranged differently for x86 and NEON
# code. This is because NEON uses a completely different code path for
# continuation byte testing, and thus we can make different tradeoffs that are
# optimized for each ISA. Namely, for AVX2/SSE4 testing the high bit of each
# byte in a vector is cheap with the vpmovmsk instruction--other bits require
# a shift to get them to the top position (or an add to test the bit in the
# next-to-highest position).
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
# For NEON, we have a different constraint: we want to keep the continuation
# byte marker in the low bit, for a rather arcane reason. We merge the
# 3/4-byte markers with two instructions, after shifting the vector lanes
# forward to line up the leader bytes with the third and fourth byte. This is
# possible by using the vector shift-right-and-accumulate instruction, which
# is safe only if we don't need to worry about the other bits overflowing and
# messing up our mask. So, by using MARK_CONT==1<<0, we can shift the 3/4-byte
# markers into the low position and add them together, resulting in the XOR of
# the two markers in the low bit. This only differs from OR when we have a
# 3-byte leader followed by a 4-byte leader, which will already be detected as
# illegal using the ERR_CONT bit.
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
# 16-entry tables. We are looking up 12 bits total, which can be done with two
# vpermb's on AVX-512. Instead of splitting the 12 bits down the middle, which
# is perhaps the more obvious approach, we combine the first two and last four
# bits of the 12 contiguous bits for the first index, and the remaining bits
# for the second index. This is illustrated below:
#
#   nibble:         1111 2222 3333
#   index 16:       xxxx yyyy zzzz
#   bad index 64:   xxxx xxyy yyyy
#   good index 64:  xxyy yyyy xxxx
#
# This is done for three reasons:
# 1) The first scheme does not have independence between the two halves of bits.
#    Namely, the ERR_MAX1 and ERR_MAX2 bits need to distinguish between these
#    cases:
#       nibbles  indices
#       0xF48    [1111 01] [00 1000]  <-- legal
#       0xF49    [1111 01] [00 1001]  <-- illegal
#       0xF58    [1111 01] [01 1000]  <-- illegal
#       0xF59    [1111 01] [01 1001]  <-- illegal
#       0xF88    [1111 10] [00 1000]  <-- illegal
#       0xF89    [1111 10] [00 1001]  <-- illegal
#    In other words, detecting 0xF88 as an error requires ERR_MAX2 to be set
#    for the second index [00 1000], but that bit set would make 0xF48 illegal
#    as well. This could be solved by separating ERR_MAX2 into two bits, for
#    the 0xF5..0xF7 and 0xF8..0xFF cases, which would be independent between
#    the two indices. But, we don't have any bits to spare!
# 2) Using the second scheme allows us to save a vector shift instruction--
#    the index bits need to be in the low 6 bits of each vector lane for the
#    vpermb instruction, and this way the second index bits are already in
#    place.
# 3) As described above, the MARK_CONT bit needs to detect continuation bytes
#    after the first. For 16-byte lookups, we can AND the first and third nibble
#    masks together, which we were doing anyways. For the two 64-byte lookups,
#    we don't get an intermediate result like that. But, combining the first
#    two and last four bits from the 12 gets us the high two bits from each
#    pair of bytes. The first two bits of a byte are enough to identify all
#    continuation bytes: they are always of the form 10xxxxxx. So, this lookup
#    will already give us the continuation-bytes-after-the-first mask, without
#    doing any extra masking.
#
# There is one downside to this approach, though: since we split the bits of
# the first nibble, we can't hide the 3/4-byte marker bits in the first lookup
# table, and need separate compare instructions. This isn't a huge deal, since
# both the compare and hypothetical bit-test are one instruction a piece, with
# the compare encoding being just a single byte longer (7 vs 6 bytes each). On
# the other hand, the compare allows us to save a single register since we
# use the same 0xF0 constant for finding 4-byte leader bytes and for the ternary
# logic instruction. So, a very tiny difference either way, but that's the kind
# of crazy stuff that matters for super-optimized code...
def make_64_bit_tables(error_bits, bit_map):
    error_bits_64 = [[0] * 64 for i in range(2)]
    for n in range(64):
        n_1 = n_2 = 0
        for x in range(4):
            n_1 |= error_bits[0][(n >> 4) | (x << 2)] & error_bits[1][n & 0xF]
            n_2 |= error_bits[2][n & 0xF] & error_bits[0][(n & 0x30) >> 2 | x]
        error_bits_64[0][n] = n_1
        error_bits_64[1][n] = n_2

    # Sanity check: splitting the indices up like we do should retain
    # independence between the indices, which means we shouldn't lose any
    # information. Make sure we can round-trip back to 16-entry tables.
    e_rt = [[0] * 16 for i in range(3)]
    for n in range(64):
        # Reconstrucing the first nibble is weird, since the bits are divided.
        # We loop over all possible values of both 64-bit indices, and set all
        # the bits that are set in both entries for all possible values that
        # correspond to each 4-bit index of the first nibble.
        for x in range(64):
            index_1 = (n >> 2) & 0xC | (x >> 4) & 0x3
            index_2 = (n >> 4) & 0x3 | (x >> 2) & 0xC
            e_rt[0][index_1] |= error_bits_64[1][n] & error_bits_64[0][x]
            e_rt[0][index_2] |= error_bits_64[0][n] & error_bits_64[1][x]
            # Err, well strictly speaking, this isn't a true round trip, due
            # to the MARK_CONT bit. Since this isn't an actual error bit, it
            # will never be set after the above ANDs. So, hackily add it back
            # here for every value of the second 64-entry index that has it.
            cont_bit = 1 << bit_map['MARK_CONT']
            e_rt[0][index_1] |= error_bits_64[1][n] & cont_bit

        e_rt[1][n & 0xF] |= error_bits_64[0][n]
        e_rt[2][n & 0xF] |= error_bits_64[1][n]

    assert error_bits == e_rt

    return error_bits_64

# Write a table to the output file, along with #defines for the actual bit
# numbers of the various error bits
def write_table(f, table, bit_map):
    t_len = len(table[0])
    step = 8 if t_len == 64 else 4
    for n in range(len(table)):
        t = ',\n    '.join(', '.join('0x%02X' % v for v in table[n][x:x+step])
                for x in range(0, t_len, step))
        f.write('const vec_t error_%s = V_TABLE_%s(\n    %s\n);\n' %
                (n+1, t_len, t))
    for k in sorted(bit_map):
        f.write('#   undef %s\n' % k)
    for k, v in sorted(bit_map.items()):
        f.write('#   define %-20s %s\n' % (k, v))

def main(path):
    x86_table = make_bit_tables(x86_bit_map)
    avx512_table = make_64_bit_tables(x86_table, x86_bit_map)
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
