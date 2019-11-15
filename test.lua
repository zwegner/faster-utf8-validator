-- Load the library
local ffi = require('ffi')
local lib_avx2 = ffi.load('_out/avx2/rel/zval.so')
local lib_sse4 = ffi.load('_out/sse4/rel/zval.so')
ffi.cdef([[
bool z_validate_utf8_avx2(const char *data, ssize_t len);
bool z_validate_utf8_sse4(const char *data, ssize_t len);
]])

local VALIDATORS = {
    lib_avx2.z_validate_utf8_avx2,
    lib_sse4.z_validate_utf8_sse4
}

-- Ranges for certain kinds of bytes
local ANY = { 0, 0xFF }
local ASCII = { 0, 0x7F }
local CONT = { 0x80, 0xBF }

-- Test cases. Format is { expected-result, byte-ranges... } where byte-ranges
-- are 2-element tables { lo, hi }. For each byte, all byte values between the
-- corresponding lo and hi values are tested.
local TEST_CASES = {
    -- ASCII. First byte is ' ' for keeping combinatorial explosions down
    {  true, { 0x20, 0x20 }, ASCII, ASCII, ASCII },

    -- 2-byte sequences
    { false, { 0xC2, 0xC2 }, },
    { false, { 0xC2, 0xC2 }, ASCII },
    {  true, { 0xC2, 0xC2 }, CONT },
    { false, { 0xC2, 0xC2 }, { 0xC0, 0xFF} },
    { false, { 0xC2, 0xC2 }, CONT, CONT },
    { false, { 0xC2, 0xC2 }, CONT, CONT, CONT },

    -- 3-byte sequences
    { false, { 0xE1, 0xE1 }, },
    { false, { 0xE1, 0xE1 }, CONT },
    {  true, { 0xE1, 0xE1 }, CONT, CONT },
    {  true, { 0xE1, 0xE1 }, CONT, CONT, ASCII },
    { false, { 0xE1, 0xE1 }, CONT, ASCII },
    { false, { 0xE1, 0xE1 }, CONT, CONT, CONT },

    -- 4-byte sequences
    { false, { 0xF1, 0xF1 }, },
    { false, { 0xF1, 0xF1 }, CONT },
    { false, { 0xF1, 0xF1 }, CONT, CONT },
    {  true, { 0xF1, 0xF1 }, CONT, CONT, CONT },
    { false, { 0xF1, 0xF1 }, CONT, CONT, ASCII },
    {  true, { 0xF1, 0xF1 }, CONT, CONT, CONT, ASCII },

    -- No C0/C1 bytes
    { false, { 0xC0, 0xC1 }, ANY },
    { false, { 0xC0, 0xC1 }, ANY, ANY },
    { false, { 0xC0, 0xC1 }, ANY, ANY, ANY },

    -- No E0 followed by 80..9F
    { false, { 0xE0, 0xE0 }, { 0x00, 0x9F }, CONT },
    {  true, { 0xE0, 0xE0 }, { 0xA0, 0xBF }, CONT },

    -- No surrogate pairs
    {  true, { 0xE1, 0xEC }, CONT, CONT },
    {  true, { 0xED, 0xED }, { 0x80, 0x9F }, CONT },
    { false, { 0xED, 0xED }, { 0xA0, 0xBF }, CONT },
    {  true, { 0xEE, 0xEF }, CONT, CONT },

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

local count, fails = 0, 0
for idx, test in ipairs(TEST_CASES) do
    local expected = table.remove(test, 1)
    local lo_1, hi_1 = unpack(table.remove(test, 1))

    -- Loop through various frame shifts, to make sure we catch any issues due
    -- to vector alignment
    for _, k in ipairs{1, 10, 28, 29, 20, 31, 32, 33} do
        local buffer = {}
        for j = 1, 64 do buffer[j] = 0 end

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

                -- Run the validator
                local str = ffi.string(string.char(unpack(buffer)), 64)
                for _, validate in ipairs(VALIDATORS) do
                    local result = validate(str, 64)
                    if result ~= expected then
                        fails = fails + 1
                        print('failure:', idx, result, expected, astr(buffer))
                        assert(false)
                    end
                    count = count + 1
                end
            end
        end

        -- Make sure we're running tests
        assert(count > last_count)
    end
end

print(('passed %d/%d tests'):format(count - fails, count))
