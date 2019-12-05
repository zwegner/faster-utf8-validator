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

#include <stdio.h>

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

#define DEBUG(x)            //x

#if defined(ASCII_CHECK)
#   define ASCII            _ascii
#else
#   define ASCII
#endif

#define NAME_(name, suff, ascii)    name##_##suff##ascii
#define NAME__(name, suff, ascii)   NAME_(name, suff, ascii)
#define NAME(name)                  NAME__(name, SUFFIX, ASCII)

#if defined(AVX2)

////////////////////////////////////////////////////////////////////////////////
// AVX2 ////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

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
#   define v_add            _mm256_add_epi32
#   define v_shl(x, shift)  ((shift) ? _mm256_slli_epi16((x), (shift)) : (x))
#   define v_shr(x, shift)  ((shift) ? _mm256_srli_epi16((x), (shift)) : (x))

#   define v_test_any(x)    !_mm256_testz_si256((x), (x))
#   define v_test_bit(input, bit)                                           \
        _mm256_movemask_epi8(v_shl((input), 7 - (bit)))

// Parallel table lookup for all bytes in a vector. We need to AND with 0x0F
// for the lookup, because vpshufb has the neat "feature" that negative values
// in an index byte will result in a zero.

#   define v_lookup(table, index, shift)                                    \
        _mm256_shuffle_epi8((table), v_and(v_shr((index), (shift)),         \
                    v_set1(0x0F)))

// Simple macro to make a vector lookup table for use with vpshufb. Since
// AVX2 is two 16-byte halves, we duplicate the input values.

#   define V_TABLE_16(...)    _mm256_setr_epi8(__VA_ARGS__, __VA_ARGS__)

// Move all the bytes in "input" to the left by one and fill in the first
// byte with zero. Since AVX2 generally works on two separate 16-byte
// vectors glued together, this needs two steps. The permute2x128 takes the
// middle 32 bytes of the 64-byte concatenation v_zero:input. The align
// then gives the final result in each half:
//      top half: input_L:input_H --> input_L[15]:input_H[0:14]
//   bottom half:  zero_H:input_L -->  zero_H[15]:input_L[0:14]
static inline vec_t NAME(v_shift_lanes)(vec_t bottom, vec_t top, uint32_t n) {
    vec_t shl_16 = _mm256_permute2x128_si256(top, bottom, 0x03);
    return _mm256_alignr_epi8(top, shl_16, 16 - n);
}

static inline vec_t NAME(v_load_shift_first)(const char *data) {
    return NAME(v_shift_lanes)(v_set1(0), v_load(data), 1);
}

#elif defined(AVX512_VBMI)

////////////////////////////////////////////////////////////////////////////////
// AVX512 //////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#   define SUFFIX           avx512_vbmi

#   define V_LEN            (64)

// Double mask struct. We manage the low and high halves manually since
// LLVM/GCC were generating crappy code for uint128_t.
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
#   define v_shr(x, shift)  _mm512_srli_epi16((x), (shift))

#   define v_test_any(x)    _mm512_test_epi8_mask((x), (x))
#   define v_test_bit(input, bit)                                           \
        _mm512_test_epi8_mask((input), v_set1((uint32_t)1 << (bit)))

#   define v_lookup_64(table, index)                                    \
        _mm512_permutexvar_epi8((index), (table))

// Same macro as for AVX2, but repeated four times

#   define V_TABLE_16(...)    _mm512_setr_epi8(__VA_ARGS__, __VA_ARGS__, \
        __VA_ARGS__, __VA_ARGS__)

#   define V_TABLE_64(...)    _mm512_setr_epi8(__VA_ARGS__)

// Hack around setr_epi8 not being available
#   define _mm512_setr_epi8(...) \
        (__extension__ (__m512i)(__v64qi) { __VA_ARGS__ } )

static inline vec_t NAME(v_shift_lanes)(vec_t bottom, vec_t top, uint32_t n) {
    return _mm_alignr_epi8(bottom, top, 16 - n);
}

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

////////////////////////////////////////////////////////////////////////////////
// SSE4 definitions ////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#   define SUFFIX           sse4

#   define V_LEN            (16)

#   define vec_t            __m128i
#   define vmask_t          uint16_t
#   define vmask2_t         uint32_t

#   define v_load(x)        _mm_lddqu_si128((vec_t *)(x))
#   define v_set1           _mm_set1_epi8
#   define v_and            _mm_and_si128
#   define v_or             _mm_or_si128
#   define v_add            _mm_add_epi32
#   define v_shl(x, shift)  ((shift) ? _mm_slli_epi16((x), (shift)) : (x))
#   define v_shr(x, shift)  ((shift) ? _mm_srli_epi16((x), (shift)) : (x))

#   define v_test_any(x)    !_mm_test_all_zeros((x), (x))
#   define v_test_bit(input, bit)                                           \
        _mm_movemask_epi8(v_shl((input), (uint8_t)(7 - (bit))))

#   define v_lookup(table, index, shift)                                    \
        _mm_shuffle_epi8((table), v_and(v_shr((index), (shift)), v_set1(0x0F)))

#   define V_TABLE_16(...)  _mm_setr_epi8(__VA_ARGS__)

static inline vec_t NAME(v_shift_lanes)(vec_t bottom, vec_t top, uint32_t n) {
    return _mm_alignr_epi8(bottom, top, 16 - n);
}

static inline vec_t NAME(v_load_shift_first)(const char *data) {
    return NAME(v_shift_lanes)(v_load(data), v_set1(0), 1);
}

#elif defined(NEON)

////////////////////////////////////////////////////////////////////////////////
// NEON definitions ////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#   define SUFFIX           neon

#   define V_LEN            (16)

// For NEON, since there is no movemask-like instruction, we keep continuation
// byte errors in a regular vector. There is no vmask2_t needed.
#   define vec_t            uint8x16_t
#   define vmask_t          uint8x16_t

#   define v_load(x)        vld1q_u8((uint8_t *)(x))
#   define v_set1           vdupq_n_u8
#   define v_and            vandq_u8
#   define v_or             vorrq_u8
#   define v_shl(x, shift)  ((shift) ? vshlq_n_u8((x), (shift)) : (x))
#   define v_shr(x, shift)  ((shift) ? vshrq_n_u8((x), (shift)) : (x))

#   define v_test_any       NAME(_v_test_any)
#   define v_test_bit(input, bit)                                           \
        v_test_any(v_and((input), v_set1((uint8_t)(1 << (bit)))))

static inline uint64_t NAME(_v_test_any)(vec_t vec) {
    uint64x2_t vec_64 = (uint64x2_t)vec;
    return vgetq_lane_u64(vec_64, 0) | vgetq_lane_u64(vec_64, 1);
}

// This logic is specific for 0/4 bit shifts--the masking of the low four bits
// is only necessary if shift < 4 (and thus some high bits might be set)
#   define v_lookup(table, index, shift)                                    \
        vqtbl1q_u8((table), (shift) ? vshrq_n_u8((index), (shift)) :        \
                v_and((index), v_set1(0x0F)))

#   define V_TABLE_16(...)  ( (uint8x16_t) { __VA_ARGS__ } )

static inline vec_t NAME(v_shift_lanes)(vec_t bottom, vec_t top, uint32_t n) {
    return vextq_u8(bottom, top, 16 - n);
}

static inline vec_t NAME(v_load_shift_first)(const char *data) {
    return NAME(v_shift_lanes)(v_set1(0), v_load(data), 1);
}

#else

#   error "No valid configuration: must define one of AVX512_VBMI, " \
        "AVX2, SSE4, or NEON"

#endif

// Result struct. We combine special-case test errors and continuation byte
// errors into one result, so we can easily unroll the inner loop and only
// branch once for several vectors' worth of results.
#define result_t    NAME(_result_t)
typedef struct {
    vec_t lookup_error;
    vmask_t cont_error;
} result_t;

// Some code paths for result checking are common between all x86 ISAs and
// different on NEON. These are collected here.

#if !defined(NEON)
#   define vmask_or(a, b)       ((a) | (b))
#   define test_carry_req(r)    ((r) != 0)
#   define result_fails(r)                                                  \
            (v_test_any((r).cont_error) || (r).lookup_error != 0)
#else
#   define vmask_or         v_or
// In the NEON code paths, we keep continuation byte masks as vectors, and need
// a special test at the end of the input to make sure we weren't expecting
// more continuation bytes. This means a 4-byte sequence starting at the second
// to last position, or a 3-byte sequence starting at the last position. Since
// the continuation marker bits are already shifted forward by one (due to
// coming from the special case error lookups), the last two positions are
// sufficient to catch any expected continuation bytes at the end.
#   define test_carry_req(r)                                                 \
        v_test_any(v_and((r), V_TABLE_16(0,0,0,0,0,0,0,0,0,0,0,0,0,0,       \
                        1 << ERR_MAX2, 1 << ERR_MAX2 | 1 << ERR_SURR)))
// Mask out everything but the MARK_CONT bit from the cont_error. This is
// just hoisting out an AND from the unrolled loop that the compiler can't
// really be trusted to do itself.
#   define result_fails(r)                                                  \
            (v_test_any(v_and((r).cont_error, v_set1(1 << MARK_CONT))) ||   \
             v_test_any((r).lookup_error))
#endif

#define state_t     NAME(_state_t)
typedef struct {
    vec_t bytes;
    vec_t shifted_bytes;
    vec_t last_bytes;
    vec_t next_shifted_bytes;
} state_t;

static inline void NAME(load_first)(state_t *state, const char *data) {
    state->next_shifted_bytes = NAME(v_load_shift_first)(data);
    state->bytes = v_set1(0);
}

static inline void NAME(load_next)(state_t *state, const char *data) {
    state->last_bytes = state->bytes;
    state->bytes = v_load(data);
#if 1 || defined(USE_UNALIGNED_LOADS)
    state->shifted_bytes = state->next_shifted_bytes;
    state->next_shifted_bytes = v_load(data + V_LEN - 1);
#else
    state->shifted_bytes = NAME(v_shift_lanes)(state->last_bytes, state->bytes, 1);
#endif
}

static void UNUSED NAME(print_vec)(vec_t a) {
    char buf[V_LEN];
    *(vec_t *)buf = a;
    printf("{");
    for (uint32_t i = 0; i < V_LEN; i++)
        printf("%2x,", buf[i] & 0xff);
    printf("}\n");
}

// Validate one vector's worth of input bytes
static inline result_t NAME(z_validate_vec)(vec_t bytes, vec_t shifted_bytes,
        vmask_t *carry_req) {
    result_t result;

    // Add error masks as locals
#include "table.h"

#if defined(AVX512_VBMI)
    vmask2_t req = { *carry_req, 0 };

    // Look up error masks for the two 6-bit indices
    vec_t e_1 = v_lookup_64(error_1, shifted_bytes);
    // Bitwise select: we want to combine the top two bits from shifted_bytes
    // with the bottom four bits from bytes, which we can do with one ternary
    // logic operation. 0xCA is an 8x1 bit lookup table, equivalent to
    // (a ? c : b). Using 0xF0 as the selector (a), we select between the
    // appropriately-shifted values of shifted_bytes (b) or bytes (c).
    vec_t index_2 = _mm512_ternarylogic_epi32(v_set1(0xF0),
            v_shr(shifted_bytes, 2), v_shr(bytes, 4), 0xCA);
    vec_t e_2 = v_lookup_64(error_2, index_2);

    // Check if any bits are set in both error masks
    result.lookup_error = v_and(e_1, e_2);

    // Get a bitmask of all continuation bytes in the input. We can cheat a bit
    // (in a fun way) by hiding the bit in the error lookup tables.
    vmask_t cont = v_test_bit(e_2, MARK_CONT);

    // Find 3/4-byte leader bytes
    for (int n = 2; n <= 3; n++) {
        vmask_t set = _mm512_cmpge_epu8_mask(bytes, v_set1(0xFF << (7-n)));

        // Add shifted bits for required continuation bytes
        req.lo += set << n;
        req.hi += set >> (64 - n);
    }

    // Save required continuation bits for the next round
    *carry_req = req.hi;

    result.cont_error = (cont ^ req.lo);

#else

    // Look up error masks for three consecutive nibbles
    vec_t e_1 = v_lookup(error_1, shifted_bytes, 4);
    vec_t e_2 = v_lookup(error_2, shifted_bytes, 0);
    vec_t e_3 = v_lookup(error_3, bytes, 4);

    // Get error bits common between the first and third nibbles. This is a
    // subexpression used for ANDing all three nibbles, but is also used for
    // finding continuation bytes after the first. The continuation bit is
    // only set in this mask if both the first and third nibbles correspond to
    // continuation bytes, so we 
    vec_t e_1_3 = v_and(e_1, e_3);

    // Create the result vector with any bits are set in all three error masks
    result.lookup_error = v_and(e_1_3, e_2);

#if defined(NEON)

    vec_t shift_1 = NAME(v_shift_lanes)(*carry_req, e_1, 1);
    vec_t shift_2 = NAME(v_shift_lanes)(*carry_req, e_1, 2);
    // Shift the 3/4 byte marker bits into place for the MARK_CONT bit,
    // and add them together. This is safe for similar reasons to the +=
    // explanation in the below x86 path, and because we use MARK_CONT==1<<0
    // on NEON (so other bits in shift_1/2 don't overflow into the MARK_CONT
    // bit). We don't AND with the MARK_CONT bit here before setting the bits
    // in result.cont_error, mainly because some compilers (at least GCC 8)
    // aren't quite smart enough to hoist the AND out of the unrolled loop.
    // We instead put the AND in the final result_fails macro.
    vec_t req_3 = v_shr(shift_1, ERR_SURR - MARK_CONT);
    vec_t req_3_4 = vsraq_n_u8(req_3, shift_2, ERR_MAX2 - MARK_CONT);

    result.cont_error = veorq_u8(req_3_4, e_1_3);

    *carry_req = e_1;

#else

    // req is a mask of what bytes are required to be continuation bytes after
    // the first, and cont is a mask of the continuation bytes after the first
    vmask2_t req = *carry_req;
    vmask_t cont = v_test_bit(e_1_3, MARK_CONT);

    // Compute the continuation byte mask by finding bytes that start with
    // 11x, 111x, and 1111. For each of these prefixes, we get a bitmask
    // and shift it forward by 1, 2, or 3. This loop should be unrolled by
    // the compiler, and the (n == 1) branch inside eliminated.
    vmask_t leader_3 = v_test_bit(e_1, ERR_SURR);
    vmask_t leader_4 = v_test_bit(v_add(e_1, e_1), ERR_MAX2+1);

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
    req += (vmask2_t)leader_3 << 1;
    req += (vmask2_t)leader_4 << 2;

    // Save continuation bits and input bytes for the next round
    *carry_req = req >> V_LEN;

    // Check that continuation bytes match. We must cast req from vmask2_t
    // (which holds the carry mask in the upper half) to vmask_t, which
    // zeroes out the upper bits
    result.cont_error = (cont ^ (vmask_t)req);

#endif

#endif

    return result;
}

int NAME(z_validate_utf8)(const char *data, size_t len) {
    // Keep continuation bits from the previous iteration that carry over to
    // each input chunk vector
    vmask_t carry_req = v_set1(0);

    size_t offset = 0;
    // Deal with the input up until the last section of bytes
    if (len >= V_LEN) {
        // We need a vector of the input byte stream shifted forward one byte.
        // Since we don't want to read the memory before the data pointer
        // (which might not even be mapped), for the first chunk of input just
        // use vector instructions.
        state_t state[1];
        NAME(load_first)(state, data);

#define UNROLL_COUNT    (8)
#define UNROLL_SIZE     (UNROLL_COUNT * V_LEN)

        for (; offset + UNROLL_SIZE + V_LEN - 1 < len; offset += UNROLL_SIZE) {
            vec_t byte_vecs[UNROLL_COUNT];
            for (uint32_t i = 0; i < UNROLL_COUNT; i++)
                byte_vecs[i] = v_load(data + offset + i * V_LEN);

#if defined(ASCII_CHECK)
            // Quick skip for ASCII-only input. If there are no bytes with the
            // high bit set, we can skip this chunk. If we expected any
            // continuation bytes here, we return invalid, otherwise just skip.
            vec_t ascii_vec = byte_vecs[0];
            for (uint32_t i = 1; i < UNROLL_COUNT; i++)
                ascii_vec = v_or(ascii_vec, byte_vecs[i]);

            if (LIKELY(!v_test_bit(ascii_vec, 7))) {
                if (test_carry_req(carry_req))
                    return 0;

                // Set up the state for the next iteration by loading the last
                NAME(load_next)(state, data + offset + (UNROLL_COUNT-1) * V_LEN);
                continue;
            }
#endif

            // Load all the vectors for the unrolled chunk
            vec_t sh_byte_vecs[UNROLL_COUNT];
            for (uint32_t i = 0; i < UNROLL_COUNT; i++) {
                NAME(load_next)(state, data + offset + i * V_LEN);
                byte_vecs[i] = state->bytes;
                sh_byte_vecs[i] = state->shifted_bytes;
            }

            // Run other validations. Annoyingly, at least one compiler (GCC 8)
            // doesn't optimize v_or(0, x) into x, so manually unroll the first
            // iteration
            result_t result = NAME(z_validate_vec)(byte_vecs[0],
                    sh_byte_vecs[0], &carry_req);
            for (uint32_t i = 1; i < UNROLL_COUNT; i++) {
                result_t r = NAME(z_validate_vec)(byte_vecs[i],
                        sh_byte_vecs[i], &carry_req);
                result.lookup_error = v_or(result.lookup_error, r.lookup_error);
                result.cont_error = vmask_or(result.cont_error, r.cont_error);
            }

            if (UNLIKELY(result_fails(result)))
                return 0;
        }

        // Loop over input in V_LEN-byte chunks, as long as we can safely read
        // that far into memory
        for (; offset + V_LEN < len; offset += V_LEN) {
            NAME(load_next)(state, data + offset);

            result_t result = NAME(z_validate_vec)(state->bytes,
                    state->shifted_bytes, &carry_req);
            if (UNLIKELY(result_fails(result)))
                return 0;
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

        vec_t bytes = v_load(buffer + 1);
        vec_t shifted_bytes = v_load(buffer);
        result_t result = NAME(z_validate_vec)(bytes, shifted_bytes,
                &carry_req);
        if (UNLIKELY(result_fails(result)))
            return 0;
    }

    // Micro-optimization compensation! We have to double check
    // for a multi-byte sequence that starts on the last byte, since we
    // check for the first continuation byte using error masks,
    // which are shifted one byte forward in the data stream. Thus, a leader
    // byte in the last position will be ignored if it's also the last byte
    // of a vector.
    if (len > 0 && (uint8_t)data[len - 1] >= 0xC0)
        return 0;

    // The input is valid if we don't have any more expected continuation bytes
    return !test_carry_req(carry_req);
}

// Undefine all macros

#undef NAME
#undef ASCII
#undef SUFFIX
#undef V_LEN
#undef vec_t
#undef vmask_t
#undef vmask2_t
#undef v_load
#undef v_set1
#undef v_and
#undef v_or
#undef v_add
#undef v_shl
#undef v_shr
#undef v_test_any
#undef v_test_bit
#undef v_lookup
#undef V_TABLE_16
#undef V_TABLE_64
#undef vmask_or
#undef test_carry_req
