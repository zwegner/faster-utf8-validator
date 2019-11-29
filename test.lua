-- Load the library
local ffi = require('ffi')
local lib_avx512 = ffi.load('_out/avx512/rel/zval.so')
local lib_avx2 = ffi.load('_out/avx2/rel/zval.so')
local lib_sse4 = ffi.load('_out/sse4/rel/zval.so')
--local lib_neon = ffi.load('_out/neon/rel/zval.so')
ffi.cdef([[
bool z_validate_utf8_avx512_vbmi(const char *data, size_t len);
bool z_validate_utf8_avx2(const char *data, size_t len);
bool z_validate_utf8_sse4(const char *data, size_t len);
bool z_validate_utf8_neon(const char *data, size_t len);
]])

local VALIDATORS = {
    ['avx512'] = lib_avx512.z_validate_utf8_avx512_vbmi,
    ['avx2']   = lib_avx2.z_validate_utf8_avx2,
    ['sse4']   = lib_sse4.z_validate_utf8_sse4,
--    ['neon']   = lib_neon.z_validate_utf8_neon,
}

-- Ranges for certain kinds of bytes
local ANY = { 0, 0xFF }
local ASCII = { 0, 0x7F }
local CONT = { 0x80, 0xBF }

-- Test cases. Format is { expected-result, byte-ranges... } where byte-ranges
-- are 2-element tables { lo, hi }. For each byte, all byte values between the
-- corresponding lo and hi values are tested.
local TEST_CASES = {
    -- ASCII
    {  true, ASCII, ASCII, ASCII, ASCII },

    -- 2-byte sequences
    { false, { 0xC2, 0xDF }, },
    { false, { 0xC2, 0xDF }, ASCII },
    {  true, { 0xC2, 0xDF }, CONT },
    { false, { 0xC2, 0xDF }, { 0xC0, 0xFF} },
    { false, { 0xC2, 0xDF }, CONT, CONT },
    { false, { 0xC2, 0xDF }, CONT, CONT, CONT },

    -- 3-byte sequences
    { false, { 0xE1, 0xE1 }, },
    { false, { 0xE1, 0xE1 }, CONT },
    {  true, { 0xE1, 0xE1 }, CONT, CONT },
    {  true, { 0xE1, 0xE1 }, CONT, CONT, ASCII },
    { false, { 0xE1, 0xE1 }, CONT, ASCII },
    { false, { 0xE1, 0xE1 }, CONT, CONT, CONT },

    -- 4-byte sequences
    { false, { 0xF1, 0xF3 }, },
    { false, { 0xF1, 0xF3 }, CONT },
    { false, { 0xF1, 0xF3 }, CONT, CONT },
    {  true, { 0xF1, 0xF3 }, CONT, CONT, CONT },
    { false, { 0xF1, 0xF3 }, CONT, CONT, ASCII },
    {  true, { 0xF1, 0xF3 }, CONT, CONT, CONT, ASCII },
    { false, { 0xF1, 0xF3 }, CONT, CONT, CONT, CONT },

    -- No C0/C1 bytes (overlong)
    { false, { 0xC0, 0xC1 }, ANY },
    { false, { 0xC0, 0xC1 }, ANY, ANY },
    { false, { 0xC0, 0xC1 }, ANY, ANY, ANY },

    -- No E0 followed by 80..9F (overlong)
    { false, { 0xE0, 0xE0 }, { 0x00, 0x9F }, CONT },
    {  true, { 0xE0, 0xE0 }, { 0xA0, 0xBF }, CONT },

    -- No surrogate pairs
    {  true, { 0xE1, 0xEC }, CONT, CONT },
    {  true, { 0xED, 0xED }, { 0x80, 0x9F }, CONT },
    { false, { 0xED, 0xED }, { 0xA0, 0xBF }, CONT },
    {  true, { 0xEE, 0xEF }, CONT, CONT },

    -- No F0 followed by 80..8F (overlong)
    { false, { 0xF0, 0xF0 }, { 0x80, 0x8F }, CONT, CONT },
    {  true, { 0xF0, 0xF0 }, { 0x90, 0xBF }, CONT, CONT },

    -- No code points above U+10FFFF
    {  true, { 0xF4, 0xF4 }, { 0x80, 0x8F }, CONT, CONT },
    { false, { 0xF4, 0xF4 }, { 0x90, 0xBF }, CONT, CONT },

    -- No bytes above F4
    { false, { 0xF5, 0xFF }, ANY },
    { false, { 0xF5, 0xFF }, ANY, ANY },
    { false, { 0xF5, 0xFF }, ANY, ANY, ANY },
}

-- Array string
function astr(array)
    local r = '{'
    for _, value in ipairs(array) do
        r = r .. ('%2X'):format(value) .. ','
    end
    return r .. '}'
end

-- A little helper function for running an input on each validator
function test_validators(str, len, buffer, expected, count, fails)
    for name, validate in pairs(VALIDATORS) do
        local result = validate(str, len)
        if result ~= expected then
            fails = fails + 1
            print('failure:', name, result, expected, astr(buffer))
            assert(false)
        end
        count = count + 1
    end
    return count, fails
end

local count, fails = 0, 0
for idx, test in ipairs(TEST_CASES) do
    local expected = table.remove(test, 1)
    local lo_1, hi_1 = unpack(table.remove(test, 1))
    -- Loop through various frame shifts, to make sure we catch any issues due
    -- to vector alignment
    --for _, k in ipairs{60, 61, 62, 63, 64, 65} do
    --for _, k in ipairs{1, 10, 28, 29, 20, 31, 32, 33, 60, 61, 62, 63, 64, 65} do
    for k = 1, 70 do
        local buffer = {}
        for j = 1, 256 do buffer[j] = 0 end

        local last_count = count

        -- Loop through first byte
        for b = lo_1, hi_1 do
            buffer[k] = b

            -- Find maximum range of values in remaining bytes
            for offset = 0, 255 do
                local any_valid = false
                for i, range in ipairs(test) do
                    i = i + k
                    local lo_2, hi_2 = unpack(range)
                    buffer[i] = lo_2 + offset
                    if buffer[i] > hi_2 then
                        buffer[i] = hi_2
                    else
                        any_valid = true
                    end
                end
                -- Break if we've run through the range of all bytes
                if #test > 0 and not any_valid then
                    break
                end

                -- Run the validators
                local str = ffi.string(string.char(unpack(buffer)), #buffer)
                count, fails = test_validators(str, #buffer, buffer, expected,
                        count, fails)
            end
        end

        -- Make sure we're running tests
        assert(count > last_count)
    end
end

-- Test that we're correctly dealing with input lengths, by feeding buffers
-- with invalid bytes before and after the given range
local TRAILING_TESTS = {
    {  true, },
    {  true, 0x40 },
    {  true, 0xC2, 0x80 },
    {  true, 0xE0, 0xA0, 0x80 },
    {  true, 0xE1, 0x80, 0x80 },
    {  true, 0xED, 0x80, 0x80 },
    {  true, 0xF4, 0x8F, 0x80, 0x80 },
    { false, 0xC2, },
    { false, 0xE1, 0x80 },
    { false, 0xF4, 0x80, 0x80 },
}

for _, test in ipairs(TRAILING_TESTS) do
    local expected = table.remove(test, 1)
    for pre = 0, 80 do
        for post = 0, 80 do
            local buffer = {}
            local len = pre + #test + post
            -- Fill in invalid bytes everywhere
            for j = 1, 256 do buffer[j] = 0xFF end
            -- Fill in valid bytes in the range being tested
            for j = 2, len+1 do buffer[j] = 0x20 end
            -- Fill in the test sequence
            for j = 1, #test do buffer[1+pre+j] = test[j] end

            local _str = ffi.string(string.char(unpack(buffer)), #buffer)
            local str = ffi.cast('const char *', _str) + 1
            count, fails = test_validators(str, len, buffer, expected,
                    count, fails)
        end
    end
end

print(('passed %d/%d tests'):format(count - fails, count))
