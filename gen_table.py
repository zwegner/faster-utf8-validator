#!/usr/bin/env python3
# faster-utf8-validator
#
# Copyright (c) 2019 Zach Wegner
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys

# Error nibble table. This table contains all the special case errors in UTF-8
# bytes, specified by the values of the first, second, and third nibbles of each
# invalid sequence (as described in z_validate.c). The entries have a friendly
# name of an error bit (which is mapped differently for different architectures)
# and the values of the three nibbles. Each nibble can either be a single value
# or a range of values.
error_nibbles = [
    # First continuation byte errors. Any byte following 0xCx..0xFx must be
    # in the range 0x80..0xBF
    ['ERR_CONT',  [[0xC, 0xF], [0x0, 0xF], [0x0, 0x7]]],
    ['ERR_CONT',  [[0xC, 0xF], [0x0, 0xF], [0xC, 0xF]]],

    # All the "normal" special case errors
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
    #
    # The above explanation is slightly different in the case of AVX-512. There,
    # we only have two lookups, so we can't use the intermediate result from the
    # first and third nibble. Luckily, as described below, we organize the
    # tables to get the continuation bit for free on AVX-512 as well.
    ['MARK_CONT', [[0x8, 0xB], [0x0,  -1], [0x8, 0xB]]],

    # ...and if that wasn't a complicated enough explanation, there is one more
    # crazy trick here to take into account. One big problem with skipping the
    # first continuation byte check is making sure that stray continuation bytes
    # are found invalid (special thanks to @jkeiser on github for pointing this
    # out, since I somehow forgot to consider it). It doesn't seem possible to
    # fit the ASCII->CONT error (e.g. 0x20 0x80) cleanly into this table; there
    # are no bits to spare, and we can't hide/overlap the relevant bits anywhere
    # else here. But, there is a saving grace: the (v)pshufb instruction that we
    # use to do these vector table lookups has a usually-annoying feature: if
    # the high bit is set on an input byte, the output byte is set to zero. We
    # normally need to mask with 0x0F to get around this. But due to a happy
    # coincidence, we can use the (v)pandn instruction to flip the input bits
    # before ANDing, and we can change the mask to 0x8F, just for the second
    # nibble (we also need to reverse the lookup table for the second nibble to
    # match the NOT). What this does is make ASCII input bytes get all zeroes
    # for the second error lookup--but we want *more* errors for ASCII, not
    # less. So we need to flip the output bits too--both in the lookup table
    # values and in the result, which we get for free by changing another
    # (v)pand to (v)pandn. Luckily, this lines up perfectly with the MARK_CONT
    # bit: for this bit, the second nibble is normally never set (since the
    # bit is just used as a marker), and the third nibble is only set for
    # continuation bytes. With the high-bit lookup zeroing, though, the second
    # nibble then becomes all ones with an ASCII input byte, making this bit
    # actually trigger an error. We have more luck in that this is the only
    # error with ASCII in the first nibble, so that this particular bit is the
    # only bit that will be affected by the high-bit zeroing--without this
    # trick, the first half of the first nibble table is all zeroes. So anyways,
    # the point is that fixing the ASCII->cont. byte problem can be done almost
    # for free: we need one extra register to hold the 0x8F constant, but we
    # save a couple bytes in instruction encoding (vpandn is one byte shorter
    # than vpand). For NEON, we don't get off quite so easily, since there's
    # no AND NOT instruction. Shucks.
    #
    # We give a different name to this bit error (MARK_CONT2), but it overlaps
    # with MARK_CONT for x86/NEON--they're the same bit. But, that won't work
    # for AVX-512! One of the two 64-entry lookup tables can see the top two
    # bits of each of two consecutive bytes, all in one index, so we can
    # perfectly detect the ASCII->continuation byte case. But, we need this bit
    # to either be set or not in the *other* table based on the first byte being
    # ASCII or not. That is, MARK_CONT should never be set in the other table,
    # but MARK_CONT2 always should be. So we need to pull out yet another trick
    # for AVX-512, which is a simple one: we make MARK_CONT2 overlap with a
    # different bit, ERR_CONT. This bit has the nice property that it's set for
    # every value of the first index (which contains the low six bits of the
    # first byte). MARK_CONT and MARK_CONT2 can thus be discriminated in the
    # lookup tables, and trigger errors properly.
    ['MARK_CONT2', [[0x0, 0x7], [0x0,  -1], [0x8, 0xB]]],
]

# Bit maps: the different error bits are arranged differently for x86 and NEON
# code. This is because NEON uses a completely different code path for
# continuation byte testing, and thus we can make different tradeoffs that are
# optimized for each ISA.

# For AVX2/SSE4 testing the high bit of each byte in a vector is cheap with the
# vpmovmskb instruction--other bits require a shift to get them to the top
# position (or an add, if we're testing the second-highest bit).
x86_bit_map = {
    'ERR_CONT':   0,
    'ERR_OVER1':  1,
    'ERR_OVER2':  2,
    'ERR_SURR':   3,
    'ERR_OVER3':  4,
    'ERR_MAX1':   5,
    'ERR_MAX2':   6,
    'MARK_CONT':  7,
    'MARK_CONT2': 7
}

# The AVX-512 bit mapping is almost identical, save for one difference: here,
# the MARK_CONT2 bit overlaps with ERR_CONT instead of MARK_CONT. As described
# above, we need a slightly different trick for AVX-512 since we aren't using
# the same vpshufb/AND sequence.
avx512_bit_map = x86_bit_map.copy()
avx512_bit_map['MARK_CONT2'] = avx512_bit_map['ERR_CONT']

# For NEON, we have a different constraint: we want to keep the continuation
# byte marker in the low bit, for a rather arcane reason. We merge the
# 3/4-byte markers with two instructions, after shifting the vector lanes
# forward to line up the leader bytes with the third and fourth byte. This is
# possible by using the vector shift-right-and-accumulate instruction, which
# is safe only if we don't need to worry about the other bits overflowing and
# messing up our mask. So, by making MARK_CONT the low bit, we can shift the
# 3/4-byte markers into the low position and add them together, resulting in the
# XOR of the two markers in the low bit. This only differs from OR when we have
# a 3-byte leader followed by a 4-byte leader, which will already be detected as
# illegal using the ERR_CONT bit. Besides MARK_CONT being 0, the ordering of the
# other bits is not constrained in any way.
neon_bit_map = {
    'ERR_CONT':   1,
    'ERR_OVER1':  2,
    'ERR_OVER2':  3,
    'ERR_SURR':   4,
    'ERR_OVER3':  5,
    'ERR_MAX1':   6,
    'ERR_MAX2':   7,
    'MARK_CONT':  0,
    'MARK_CONT2': 0
}

def make_bit_tables(bit_map, flip_second_nibble=False):
    # Transform single values to [n, n] ranges, and remap bits
    errors = [[bit_map[bit], [[n, n] if isinstance(n, int) else n
        for n in nibbles]] for bit, nibbles in error_nibbles]

    # Compute the three 16-entry lookup tables based on the three nibbles
    error_bits = [[0] * 16 for i in range(3)]
    for bit, nibbles in errors:
        for n, (lo, hi) in enumerate(nibbles):
            for value in range(lo, hi+1):
                error_bits[n][value] |= 1 << bit

    # Flip the second nibble if requested, as described above--for all the
    # architectures that use 16-byte lookups (i.e. everything but AVX-512), the
    # input and output bits for the second nibble table are both flipped. So we
    # need to reverse the order of the table values and invert them.
    if flip_second_nibble:
        error_bits[1] = [0xFF ^ entry for entry in error_bits[1][::-1]]

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
        # The second and third nibbles are easy: they both use the low four bits
        # of their respective six-bit indices
        e_rt[1][n & 0xF] |= error_bits_64[0][n]
        e_rt[2][n & 0xF] |= error_bits_64[1][n]
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
    # Add #defines for the names of the various bits. This is actually a lot
    # uglier than it seems: we'd really want to use an enum, but this generated
    # table.h is included locally inside our validation function. 
    for k in sorted(bit_map):
        f.write('#   undef %s\n' % k)
    for k, v in sorted(bit_map.items()):
        f.write('#   define %-20s %s\n' % (k, v))

def main(args):
    x86_table = make_bit_tables(x86_bit_map, flip_second_nibble=True)
    neon_table = make_bit_tables(neon_bit_map, flip_second_nibble=True)

    avx512_table_16 = make_bit_tables(avx512_bit_map)
    avx512_table = make_64_bit_tables(avx512_table_16, avx512_bit_map)

    # Write the output file
    output = open(args[1], 'w') if len(args) > 1 else sys.stdout
    with output as f:
        f.write('#if defined(AVX512_VBMI)\n')
        write_table(f, avx512_table, x86_bit_map)
        f.write('#elif defined(NEON)\n')
        write_table(f, neon_table, neon_bit_map)
        f.write('#else\n')
        write_table(f, x86_table, x86_bit_map)
        f.write('#endif\n')

if __name__ == '__main__':
    main(sys.argv)
