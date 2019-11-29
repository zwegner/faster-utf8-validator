// faster-utf8-validator
//
// Copyright (c) 2019 Zach Wegner
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stddef.h>
#include <stdint.h>

#if defined(NEON)
#   include <arm_neon.h>
#else
#   include <immintrin.h>
#endif

// How this validator works:
//
//   [[[ UTF-8 refresher: UTF-8 encodes text in sequences of "code points",
//   each one from 1-4 bytes. For each code point that is longer than one byte,
//   the code point begins with a unique prefix that specifies how many bytes
//   follow. All bytes in the code point after this first have a continuation
//   marker. All code points in UTF-8 will thus look like one of the following
//   binary sequences, with x meaning "don't care":
//      1 byte:  0xxxxxxx
//      2 bytes: 110xxxxx  10xxxxxx
//      3 bytes: 1110xxxx  10xxxxxx  10xxxxxx
//      4 bytes: 11110xxx  10xxxxxx  10xxxxxx  10xxxxxx
//   ]]]
//
// This validator works in two basic steps: checking continuation bytes, and
// handling special cases. Each step works on one vector's worth of input
// bytes at a time.
//
// The continuation bytes are handled in a fairly straightforward manner in
// the scalar domain. A mask is created from the input byte vector for each
// of the highest four bits of every byte. The first mask allows us to quickly
// skip pure ASCII input vectors, which have no bits set. The first and
// (inverted) second masks together give us every continuation byte (10xxxxxx).
// The other masks are used to find prefixes of multi-byte code points (110,
// 1110, 11110). For these, we keep a "required continuation" mask, by shifting
// these masks 1, 2, and 3 bits respectively forward in the byte stream. That
// is, we take a mask of all bytes that start with 11, and shift it left one
// bit forward to get the mask of all the first continuation bytes, then do the
// same for the second and third continuation bytes. Here's an example input
// sequence along with the corresponding masks:
//
//   bytes:        61 C3 80 62 E0 A0 80 63 F0 90 80 80 00
//   code points:  61|C3 80|62|E0 A0 80|63|F0 90 80 80|00
//   # of bytes:   1 |2  - |1 |3  -  - |1 |4  -  -  - |1
//   cont. mask 1: -  -  1  -  -  1  -  -  -  1  -  -  -
//   cont. mask 2: -  -  -  -  -  -  1  -  -  -  1  -  -
//   cont. mask 3: -  -  -  -  -  -  -  -  -  -  -  1  -
//   cont. mask *: 0  0  1  0  0  1  1  0  0  1  1  1  0
//
// The final required continuation mask is then compared with the mask of
// actual continuation bytes, and must match exactly in valid UTF-8. The only
// complication in this step is that the shifted masks can cross vector
// boundaries, so we need to keep a "carry" mask of the bits that were shifted
// past the boundary in the last loop iteration.
//
// Besides the basic prefix coding of UTF-8, there are several invalid byte
// sequences that need special handling. These are due to three factors:
// code points that could be described in fewer bytes, code points that are
// part of a surrogate pair (which are only valid in UTF-16), and code points
// that are past the highest valid code point U+10FFFF.
//
// All of the invalid sequences can be detected by independently observing
// the first three nibbles of each code point. Since AVX2 can do a 4-bit/16-byte
// lookup in parallel for all 32 bytes in a vector, we can create bit masks
// for all of these error conditions, look up the bit masks for the three
// nibbles for all input bytes, and AND them together to get a final error mask,
// that must be all zero for valid UTF-8. This is somewhat complicated by
// needing to shift the error masks from the first and second nibbles forward in
// the byte stream to line up with the third nibble.
//
// We have these possible values for valid UTF-8 sequences, broken down
// by the first three nibbles:
//
//   1st   2nd   3rd   comment
//   0..7  0..F        ASCII
//   8..B  0..F        continuation bytes
//   C     2..F  8..B  C0 xx and C1 xx can be encoded in 1 byte
//   D     0..F  8..B  D0..DF are valid with a continuation byte
//   E     0     A..B  E0 8x and E0 9x can be encoded with 2 bytes
//         1..C  8..B  E1..EC are valid with continuation bytes
//         D     8..9  ED Ax and ED Bx correspond to surrogate pairs
//         E..F  8..B  EE..EF are valid with continuation bytes
//   F     0     9..B  F0 8x can be encoded with 3 bytes
//         1..3  8..B  F1..F3 are valid with continuation bytes
//         4     8     F4 8F BF BF is the maximum valid code point
//
// That leaves us with these invalid sequences, which would otherwise fit
// into UTF-8's prefix encoding. Each of these invalid sequences needs to
// be detected separately, with their own bits in the error mask.
//
//   1st   2nd   3rd   error bit
//   C     0..1  0..F  0x01
//   E     0     8..9  0x02
//         D     A..B  0x04
//   F     0     0..8  0x08
//         4     9..F  0x10
//         5..F  0..F  0x20
//
// For every possible value of the first, second, and third nibbles, we keep
// a lookup table that contains the bitwise OR of all errors that that nibble
// value can cause. For example, the first nibble has zeroes in every entry
// except for C, E, and F, and the third nibble lookup has the 0x21 bits in
// every entry, since those errors don't depend on the third nibble. After
// doing a parallel lookup of the first/second/third nibble values for all
// bytes, we AND them together. Only when all three have an error bit in common
// do we fail validation.

#define LIKELY(x)           __builtin_expect((x), (1))
#define UNLIKELY(x)         __builtin_expect((x), (0))

#define UNUSED              __attribute__((unused))

#if defined(ASCII_CHECK)
#   define ASCII            _ascii
#else
#   define ASCII
#endif

#define NAME_(name, suff, ascii)    name##_##suff##ascii
#define NAME__(name, suff, ascii)   NAME_(name, suff, ascii)
#define NAME(name)                  NAME__(name, SUFFIX, ASCII)

#if defined(AVX2)

// AVX2 definitions

#   define SUFFIX           avx2

#   define V_LEN            (32)

// Vector and vector mask types. We use #defines instead of typedefs so this
// header can be included multiple times with different configurations

#   define vec_t            __m256i
#   define vmask_t          uint32_t
#   define vmask2_t         uint64_t

#   define v_load(x)        _mm256_loadu_si256((vec_t *)(x))
#   define v_set1           _mm256_set1_epi8
#   define v_and            _mm256_and_si256
#   define v_or             _mm256_or_si256
#   define v_add            _mm256_add_epi8

#   define v_test_bit(input, bit)                                           \
        _mm256_movemask_epi8(_mm256_slli_epi16((input), 7 - (bit)))

#   define v_mask_test_bit(mask, input, bit) \
        do { (mask) &= v_test_bit((input), (bit)); } while (0)

// Parallel table lookup for all bytes in a vector. We need to AND with 0x0F
// for the lookup, because vpshufb has the neat "feature" that negative values
// in an index byte will result in a zero.

#   define v_lookup(table, index, shift)                                    \
        _mm256_shuffle_epi8((table),                                        \
                v_and(_mm256_srli_epi16((index), (shift)), v_set1(0x0F)))

#   define v_test_any(x)    !_mm256_testz_si256((x), (x))

// Simple macro to make a vector lookup table for use with vpshufb. Since
// AVX2 is two 16-byte halves, we duplicate the input values.

#   define V_TABLE_16(...)    _mm256_setr_epi8(__VA_ARGS__, __VA_ARGS__)

static inline vec_t NAME(v_load_shift_first)(const char *data) {
    vec_t input = v_load(data);
    vec_t zero = v_set1(0);
    // Move all the bytes in "input" to the left by one and fill in the first
    // byte with zero. Since AVX2 generally works on two separate 16-byte
    // vectors glued together, this needs two steps. The permute2x128 takes the
    // middle 32 bytes of the 64-byte concatenation v_zero:input. The align
    // then gives the final result in each half:
    //      top half: input_L:input_H --> input_L[15]:input_H[0:14]
    //   bottom half:  zero_H:input_L -->  zero_H[15]:input_L[0:14]
    vec_t shl_16 = _mm256_permute2x128_si256(v_load(data), zero, 0x03);
    return _mm256_alignr_epi8(input, shl_16, 15);
}

#elif defined(AVX512_VBMI)

// AVX512 definitions

#   define SUFFIX           avx512_vbmi

#   define V_LEN            (64)

// Vector and vector mask types. We use #defines instead of typedefs so this
// header can be included multiple times with different configurations

typedef struct {
    __mmask64 lo, hi;
} NAME(vmask2_t);

#   define vec_t            __m512i
#   define vmask_t          __mmask64
#   define vmask2_t         NAME(vmask2_t)

#   define v_load(x)        _mm512_loadu_si512((vec_t *)(x))
#   define v_set1           _mm512_set1_epi8
#   define v_and            _mm512_and_si512
#   define v_or             _mm512_or_si512
#   define v_test_any(x)    _mm512_test_epi8_mask((x), (x))
#   define v_add            _mm512_add_epi8

#   define v_test_bit(input, bit)                                           \
        _mm512_test_epi8_mask((input), v_set1((uint32_t)1 << (bit)))

#   define v_mask_test_bit(mask, input, bit) \
        do { (mask) = _mm512_mask_test_epi8_mask((mask), (input),           \
                v_set1((uint32_t)1 << (bit))); } while (0)

#   define v_lookup(table, index, shift)                                    \
        _mm512_permutexvar_epi8((shift) ?                                   \
                _mm512_srli_epi16((index), (shift)) : (index), (table))

#   define v_lookup_64(table, index)                                    \
        _mm512_permutexvar_epi8((index), (table))

// Same macro as for AVX2, but repeated four times

#   define V_TABLE_16(...)    _mm512_setr_epi8(__VA_ARGS__, __VA_ARGS__, \
        __VA_ARGS__, __VA_ARGS__)

#   define V_TABLE_64(...)    _mm512_setr_epi8(__VA_ARGS__)

// Hack around setr_epi8 not being available
#   define _mm512_setr_epi8(...) \
        (__extension__ (__m512i)(__v64qi) { __VA_ARGS__ } )

// Load from the "data" pointer, but shifted one byte forward. We want to do
// this without touching memory before the pointer, since it might be unmapped.
// Rather than mucking around with permutes or something to do this, we can use
// a mask register to load starting from [data - 1], without actually loading
// into the first byte of the vector (which is set to zero due to the _maskz
// load variant). This will not fault if [data - 1] is invalid memory. Intel's
// docs are rather vague here, just mentioning that masked loads have "fault
// suppression", but this in fact means that lanes not in the mask cannot
// trigger page faults. See: https://stackoverflow.com/questions/54497141
static inline vec_t NAME(v_load_shift_first)(const char *data) {
    // All bits but the first
    __mmask64 shift_mask = ~1ULL;
    return _mm512_maskz_loadu_epi8(shift_mask, data - 1);
}

#elif defined(SSE4)

// SSE definitions. We require at least SSE4.1 for _mm_test_all_zeros()

#   define SUFFIX           sse4

#   define V_LEN            (16)

#   define vec_t            __m128i
#   define vmask_t          uint16_t
#   define vmask2_t         uint32_t

#   define v_load(x)        _mm_lddqu_si128((vec_t *)(x))
#   define v_set1           _mm_set1_epi8
#   define v_and            _mm_and_si128
#   define v_or             _mm_or_si128
#   define v_test_any(x)    !_mm_test_all_zeros((x), (x))
#   define v_add            _mm_add_epi8

#   define v_test_bit(input, bit)                                           \
        _mm_movemask_epi8(_mm_slli_epi16((input), (uint8_t)(7 - (bit))))

#   define v_mask_test_bit(mask, input, bit) \
        do { (mask) &= v_test_bit((input), (bit)); } while (0)

#   define v_lookup(table, index, shift)                                    \
        _mm_shuffle_epi8((table),                                           \
                v_and(_mm_srli_epi16((index), (shift)), v_set1(0x0F)))

#   define V_TABLE_16(...)  _mm_setr_epi8(__VA_ARGS__)

static inline vec_t NAME(v_load_shift_first)(const char *data) {
    return _mm_alignr_epi8(v_load(data), v_set1(0), 15);
}

#elif defined(NEON)

// NEON definitions

#   define SUFFIX           neon

#   define V_LEN            (16)

#   define vec_t            uint8x16_t
#   define vmask_t          uint32_t
#   define vmask2_t         uint64_t

#   define v_load(x)        vld1q_u8((uint8_t *)(x))
#   define v_set1           vdupq_n_u8
#   define v_and            vandq_u8
#   define v_or             vorrq_u8
#   define v_test_any       NAME(_v_test_any)

static inline uint64_t NAME(_v_test_any)(vec_t vec) {
    uint32x2_t sat_32 = vqmovn_u64(vreinterpretq_u64_u8(vec));
    return vget_lane_u64(vreinterpret_u64_u32(sat_32), 0);
}

#   define v_test_bit(input, bit)                                           \
        _mm_movemask_epi8(vshlq_n_u8((input), (uint8_t)(7 - (bit))))

#   define v_mask_test_bit(mask, input, bit) \
        do { (mask) &= v_test_bit((input), (bit)); } while (0)

#   define v_lookup(table, index, shift)                                    \
        vqtbl1q_u8((table), (shift) ? vshrq_n_u8((index), (shift)) :        \
                v_and((index), v_set1(0x0F)))

#   define V_TABLE_16(...)  ( (uint8x16_t) { __VA_ARGS__ } )

static inline vec_t NAME(v_load_shift_first)(const char *data) {
    return vextq_u8(v_set1(0), v_load(data), 15);
}

#define DEBUG(x)    //x

#include <stdio.h>
static void __attribute__((unused)) NAME(print_vec)(vec_t a) {
    char buf[V_LEN];
    *(vec_t *)buf = a;
    printf("{");
    for (uint32_t i = 0; i < V_LEN; i++)
        printf("%2x,", buf[i] & 0xff);
    printf("}\n");
}

static inline vmask2_t NAME(v_reduce_shift_6)(vec_t input) {
    const vec_t mask_8  = V_TABLE_16(-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0);
    const vec_t mask_16 = V_TABLE_16(-1,-1,0,0,-1,-1,0,0,-1,-1,0,0,-1,-1,0,0);

    uint16x8_t input_16 = (uint16x8_t)input;

    uint8x16_t sum_32_8 = (uint8x16_t)vsraq_n_u16(input_16, input_16, 6);
    uint32x4_t sum_32   = (uint32x4_t)v_and(sum_32_8, mask_8);
    uint8x16_t sum_64_8 = (uint8x16_t)vsraq_n_u32(sum_32, sum_32, 12);
    uint64x2_t sum_64   = (uint64x2_t)v_and(sum_64_8, mask_16);

    uint32x4_t lanes_32 = (uint32x4_t)vsraq_n_u64(sum_64, sum_64, 24);

    DEBUG(
        printf("reduce:\n");
        NAME(print_vec)(vreinterpretq_u8_u32(sum_32));
        NAME(print_vec)(vreinterpretq_u8_u64(sum_64));
        NAME(print_vec)(vreinterpretq_u8_u32(lanes_32));
        printf("  end=%lx\n",
            vgetq_lane_u32(lanes_32, 0) | ((uint64_t)vgetq_lane_u32(lanes_32, 2) << 16));
    )

    // note _u32 and 2 here
    return vgetq_lane_u32(lanes_32, 0) | ((uint64_t)vgetq_lane_u32(lanes_32, 2) << 16);
}

static inline vmask2_t NAME(v_reduce_shift_7)(vec_t input) {
    const vec_t mask = V_TABLE_16(-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0);

    //printf("reduce:\n");
    uint16x8_t input16 = vreinterpretq_u16_u8(input);
    // This part copy+pasted from the stack overflow answer
    uint32x4_t paired16 = vreinterpretq_u32_u16(
                              vsraq_n_u16(input16, input16, 7));
    paired16 = vreinterpretq_u32_u8(v_and(mask, vreinterpretq_u8_u32(paired16)));
    //NAME(print_vec)(vreinterpretq_u8_u32(paired16));
    uint64x2_t paired32 = vreinterpretq_u64_u32(
                              vsraq_n_u32(paired16, paired16, 14));
    //NAME(print_vec)(vreinterpretq_u8_u64(paired32));
    uint16x8_t paired64 = vreinterpretq_u16_u64(
                              vsraq_n_u64(paired32, paired32, 28));
    //NAME(print_vec)(vreinterpretq_u8_u16(paired64));
    //printf("  end=%u\n",
        //vgetq_lane_u16(paired64, 0) | ((int)vgetq_lane_u16(paired64, 4) << 8));
    // note _u16 and 4 here
    return vgetq_lane_u16(paired64, 0) | ((int)vgetq_lane_u16(paired64, 4) << 8);
}

#else

#   error "No valid configuration: must define one of AVX512_VBMI, " \
        "AVX2, SSE4, or NEON"

#endif

#undef PERM1
#undef PERM2
#undef PERM3

#if defined(AVX512_VBMI)
#define PERM1   0
#define PERM2   0
#define PERM3   1
#else
#define PERM1   0
#define PERM2   0
#define PERM3   0
#endif

#if defined(NEON)

static inline vmask_t NAME(z_validate_cont)(vec_t bytes, UNUSED vec_t shifted_bytes, vmask_t *last_cont) {
    vmask2_t req, error;
    if (0) {
        const vec_t req_table = V_TABLE_16(
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x01,
            0x02, 0x02, 0x06, 0x0E
        );

        const vec_t ones = v_set1(1);
        const vec_t valid_req = v_set1(0x0E);

        vec_t v_req = v_lookup(req_table, bytes, 4);
        vec_t v_cont = v_and(v_req, ones);
        v_req = v_and(v_req, valid_req);

        DEBUG(NAME(print_vec)(v_req);)
        DEBUG(NAME(print_vec)(v_cont);)

        req = NAME(v_reduce_shift_7)(v_req) | *last_cont;
        vmask_t cont = NAME(v_reduce_shift_7)(v_cont);
        error = (uint16_t)(cont ^ req);
        *last_cont = (uint16_t)(req >> V_LEN);
    } else {
        vec_t v_req;
        if (1) {
            const vec_t req_table = V_TABLE_16(
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00,
                0x01, 0x01, 0x01, 0x01,
                0x02, 0x02, 0x0A, 0x2A
            );
            v_req = v_lookup(req_table, bytes, 4);
        } else {
#include "table.h"
            v_req = v_lookup(error_1, bytes, 4);
            const vec_t valid_req = v_set1(0x2B);
            v_req = v_and(v_req, valid_req);
        }

        DEBUG(NAME(print_vec)(bytes);)
        DEBUG(NAME(print_vec)(v_req);)
        req = (NAME(v_reduce_shift_6)(v_req) << 1) | *last_cont;

        DEBUG(printf("req: %lx\n", req);)
        error = ((req >> 1) ^ req) & 0x55555555;
        DEBUG(printf("error: %x\n", error);)

        *last_cont = req >> V_LEN*2;
    }
    return error;
}

#elif defined(AVX512_VBMI)

static inline vmask_t NAME(z_validate_cont)(vec_t bytes, vec_t shifted_bytes, vmask_t *last_cont) {
    // Which bytes are required to be continuation bytes
    vmask2_t req = {*last_cont, 0};
    // A bitmask of the actual continuation bytes in the input
    vmask_t cont;

#include "table.h"
    // XXX this should be CSEd out
    vec_t index_2 = _mm512_ternarylogic_epi32(v_set1(0xF0),
            _mm512_srli_epi16(shifted_bytes, 2),
            _mm512_srli_epi16(bytes, 4), 0xCA);
    vec_t e_2 = _mm512_permutexvar_epi8(index_2, error_64_2);

    (void)error_64_1;
    cont = v_test_bit(e_2, 7);

    for (int n = 2; n <= 3; n++) {
        vmask_t set = _mm512_cmp_epu8_mask(bytes, v_set1(0xFF << (7-n)), _MM_CMPINT_NLT);

        // We add the shifted mask here instead of ORing it, which would
        // be the more natural operation, so that this line can be done
        // with one lea. While adding could give a different result due
        // to carries, this will only happen for invalid UTF-8 sequences,
        // and in a way that won't cause it to pass validation. Reasoning:
        // Any bits for required continuation bytes come after the bits
        // for their leader bytes, and are all contiguous. For a carry to
        // happen, two of these bit sequences would have to overlap. If
        // this is the case, there is a leader byte before the second set
        // of required continuation bytes (and thus before the bit that
        // will be cleared by a carry). This leader byte will not be
        // in the continuation mask, despite being required. QEDish.

        req.lo += set << n;
        req.hi += set >> (64-n);
    }

    // Save continuation bits and input bytes for the next round
    *last_cont = req.hi;

    return (cont ^ req.lo);
}

#else

static inline vmask_t NAME(z_validate_cont)(vec_t bytes, UNUSED vec_t shifted_bytes, vmask_t *last_cont) {
    // Which bytes are required to be continuation bytes
    vmask2_t req = *last_cont;
    // A bitmask of the actual continuation bytes in the input
    vmask_t cont;

    // Compute the continuation byte mask by finding bytes that start with
    // 11x, 111x, and 1111. For each of these prefixes, we get a bitmask
    // and shift it forward by 1, 2, or 3. This loop should be unrolled by
    // the compiler, and the (n == 1) branch inside eliminated.
    vmask_t high = v_test_bit(bytes, 7);
    vmask_t set = high;

    for (int n = 1; n <= 3; n++) {
        v_mask_test_bit(set, bytes, 7 - n);

        // Mark continuation bytes: those that have the high bit set but
        // not the next one
        if (n == 1)
            cont = high ^ set;

        // We add the shifted mask here instead of ORing it, which would
        // be the more natural operation, so that this line can be done
        // with one lea. While adding could give a different result due
        // to carries, this will only happen for invalid UTF-8 sequences,
        // and in a way that won't cause it to pass validation. Reasoning:
        // Any bits for required continuation bytes come after the bits
        // for their leader bytes, and are all contiguous. For a carry to
        // happen, two of these bit sequences would have to overlap. If
        // this is the case, there is a leader byte before the second set
        // of required continuation bytes (and thus before the bit that
        // will be cleared by a carry). This leader byte will not be
        // in the continuation mask, despite being required. QEDish.
        req += (vmask2_t)set << n;
    }

    // Save continuation bits and input bytes for the next round
    *last_cont = req >> V_LEN;

    // Check that continuation bytes match. We must cast req from vmask2_t
    // (which holds the carry mask in the upper half) to vmask_t, which
    // zeroes out the upper bits
    return (cont ^ (vmask_t)req);
}

#endif

// Validate one vector's worth of input bytes
static inline vec_t NAME(z_validate_special)(vec_t bytes, vec_t shifted_bytes) {
//    // Error lookup tables for the first, second, and third nibbles
//    const vec_t error_1 = V_TABLE_16(
//        0x00, 0x00, 0x00, 0x00,
//        0x00, 0x00, 0x00, 0x00,
//        0x00, 0x00, 0x00, 0x00,
//        0x01, 0x00, 0x06, 0x38
//    );
//    const vec_t error_2 = V_TABLE_16(
//        0x0B, 0x01, 0x00, 0x00,
//        0x10, 0x20, 0x20, 0x20,
//        0x20, 0x20, 0x20, 0x20,
//        0x20, 0x24, 0x20, 0x20
//    );
//    const vec_t error_3 = V_TABLE_16(
//        0x29, 0x29, 0x29, 0x29,
//        0x29, 0x29, 0x29, 0x29,
//        0x2B, 0x33, 0x35, 0x35,
//        0x31, 0x31, 0x31, 0x31
//    );
#include "table.h"

#if PERM2
    vmask_t high = v_test_bit(shifted_bytes, 7);

    vec_t e_1 = _mm512_maskz_permutex2var_epi8(high, error_64_1, shifted_bytes, error_64_2);
    vec_t e_2 = _mm512_permutexvar_epi8(_mm512_srli_epi16(bytes, 4), error_64_3);

    // Check if any bits are set in all three error masks
    return v_and(e_1, e_2);

#elif PERM1
    // Look up error masks for three consecutive nibbles.
    vec_t e_1 = _mm512_permutexvar_epi8(_mm512_srli_epi16(shifted_bytes, 2), error_64_1);
    // Bitwise select
    vec_t index_2 = _mm512_ternarylogic_epi32(v_set1(0xF0),
            _mm512_slli_epi16(shifted_bytes, 4),
            _mm512_srli_epi16(bytes, 4), 0xCA);
    //vec_t e_2 = _mm512_permutex2var_epi8(error_64_2, index_2, error_64_3);
    vec_t e_2 = _mm512_permutexvar_epi8(index_2, error_64_2);

    // Check if any bits are set in all three error masks
    return v_and(e_1, e_2);
#elif PERM3
    // Look up error masks for three consecutive nibbles.
    vec_t e_1 = _mm512_permutexvar_epi8(shifted_bytes, error_64_1);
    // Bitwise select
    vec_t index_2 = _mm512_ternarylogic_epi32(v_set1(0xF0),
            _mm512_srli_epi16(shifted_bytes, 2),
            _mm512_srli_epi16(bytes, 4), 0xCA);
    vec_t e_2 = _mm512_permutexvar_epi8(index_2, error_64_2);

    // Check if any bits are set in all three error masks
    return v_and(e_1, e_2);
#else
    // Look up error masks for three consecutive nibbles.
    vec_t e_1 = v_lookup(error_1, shifted_bytes, 4);
    vec_t e_2 = v_lookup(error_2, shifted_bytes, 0);
    vec_t e_3 = v_lookup(error_3, bytes, 4);

    // Check if any bits are set in all three error masks
    return v_and(v_and(e_1, e_2), e_3);
#endif
}

int NAME(z_validate_utf8)(const char *data, size_t len) {
    vec_t bytes, shifted_bytes;

    // Keep continuation bits from the previous iteration that carry over to
    // each input chunk vector
    vmask_t last_cont = 0;

    size_t offset = 0;
    // Deal with the input up until the last section of bytes
    if (len >= V_LEN) {
        // We need a vector of the input byte stream shifted forward one byte.
        // Since we don't want to read the memory before the data pointer
        // (which might not even be mapped), for the first chunk of input just
        // use vector instructions.
        shifted_bytes = NAME(v_load_shift_first)(data);

#define CHUNK_LEN       (8)
#define CHUNK_SIZE      (CHUNK_LEN * V_LEN)

    // Quick skip for ascii-only input. If there are no bytes with the high bit
    // set, we don't need to do any more work. We return either valid or
    // invalid based on whether we expected any continuation bytes here.
    //vmask_t high = v_test_bit(bytes, 7);
//    if (!high)
//        return *last_cont == 0;

        for (; offset + CHUNK_SIZE < len; offset += CHUNK_SIZE) {
            vec_t byte_vecs[CHUNK_LEN];
            vec_t sh_byte_vecs[CHUNK_LEN];
            for (uint32_t i = 0; i < CHUNK_LEN; i++) {
                const char *d = data + offset + i * V_LEN;
                byte_vecs[i] = v_load(d);
                sh_byte_vecs[i] = shifted_bytes;
                shifted_bytes = v_load(d + V_LEN - 1);
            }

#if defined(ASCII_CHECK)
            // Compute ASCII check for all vectors
            vec_t ascii_vec = v_set1(0);
            for (uint32_t i = 0; i < CHUNK_LEN; i++)
                ascii_vec = v_or(ascii_vec, byte_vecs[i]);

            if (LIKELY(!v_test_bit(ascii_vec, 7))) {
                if (last_cont != 0)
                    return 0;
                continue;
            }
#endif

            // Run other validations
            vmask_t all_c_error = 0;
            vec_t all_v_error = v_set1(0);
            for (uint32_t i = 0; i < CHUNK_LEN; i++) {
                vmask_t c_error = NAME(z_validate_cont)(byte_vecs[i], sh_byte_vecs[i], &last_cont);
                vec_t v_error = NAME(z_validate_special)(byte_vecs[i], sh_byte_vecs[i]);
                all_c_error |= c_error;
                all_v_error = v_or(all_v_error, v_error);
            }

            if (UNLIKELY(all_c_error || v_test_any(all_v_error)))
                return 0;
        }

        // Loop over input in V_LEN-byte chunks, as long as we can safely read
        // that far into memory
        for (; offset + V_LEN < len; offset += V_LEN) {
            bytes = v_load(data + offset);
            vmask_t c_error = NAME(z_validate_cont)(bytes, shifted_bytes, &last_cont);
            vec_t v_error = NAME(z_validate_special)(bytes, shifted_bytes);
            if (UNLIKELY(c_error || v_test_any(v_error)))
                return 0;
            shifted_bytes = v_load(data + offset + V_LEN - 1);
        }
    }
    // Deal with any bytes remaining. Rather than making a separate scalar path,
    // just fill in a buffer, reading bytes only up to len, and load from that.
    if (offset < len) {
        char buffer[V_LEN + 1] = { 0 };
        if (offset > 0)
            buffer[0] = data[offset - 1];
        for (int i = 0; i < (int)(len - offset); i++)
            buffer[i + 1] = data[offset + i];

        bytes = v_load(buffer + 1);
        shifted_bytes = v_load(buffer);
        vmask_t c_error = NAME(z_validate_cont)(bytes, shifted_bytes, &last_cont);
        vec_t v_error = NAME(z_validate_special)(bytes, shifted_bytes);
        if (UNLIKELY(c_error || v_test_any(v_error)))
            return 0;
    }

#if defined(AVX512_VBMI)
    if (len > 0 && (uint8_t)data[len - 1] > 0xB0)
        return 0;
#endif

    // The input is valid if we don't have any more expected continuation bytes
    return last_cont == 0;
}

// Undefine all macros

#undef NAME
#undef SUFFIX
#undef V_LEN
#undef vec_t
#undef vmask_t
#undef vmask2_t
#undef v_load
#undef v_set1
#undef v_and
#undef v_or
#undef v_test_bit
#undef v_test_any
#undef v_mask_test_bit
#undef v_lookup
#undef V_TABLE_16

#undef PERM1
#undef PERM2
