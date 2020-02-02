#!/usr/bin/env luajit

-- faster-utf8-validator
--
-- Copyright (c) 2019 Zach Wegner
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included in
-- all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

local ffi = require('ffi')

-- Parse options
local opts = {}
for i, opt in ipairs(arg) do
    if opt == '--skip' then
        opts[opt] = tonumber(arg[i + 1])
        table.remove(arg, i + 1)
    else
        opts[opt] = i
    end
end

local skip = opts['--skip'] or 0

-- Parse -b option, to build with make.py
local build = opts['-b']

-- Parse arguments to see what arches/configuration we're testing
local conf = opts['-d'] and 'deb' or 'rel'
local arches
if opts['--neon'] then
    arches = {'neon'}
else
    arches = {'avx2', 'sse4'}
    if opts['--avx512'] then
        table.insert(arches, 'avx512_vbmi')
    end
end

-- Get paths of requested libraries, so we can build if requested
local cmd = 'make.py'
local lib_paths = {}
for _, arch in ipairs(arches) do
    local path = '_out/'..arch..'/'..conf..'/zval.so'
    lib_paths[arch] = path
    cmd = cmd .. ' ' .. path
end

if build then
    ffi.cdef('int system(const char *command);')
    assert(ffi.C.system(cmd) == 0, 'build failed')
end

-- Load libraries and create a table of all validator functions we're testing
local libs = {}
local VALIDATORS = {}
for _, arch in ipairs(arches) do
    local lib = ffi.load(lib_paths[arch])
    -- XXX keep a reference to the library, apparently the function reference
    -- below isn't enough for luajit to keep the library from being gc'ed
    table.insert(libs, lib)

    local fn = 'z_validate_utf8_'..arch
    ffi.cdef('bool '..fn..'(const char *data, size_t len);')
    VALIDATORS[arch] = lib[fn]
end

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
    { false, { 0xE1, 0xEC }, },
    { false, { 0xE1, 0xEC }, CONT },
    {  true, { 0xE1, 0xEC }, CONT, CONT },
    {  true, { 0xE1, 0xEC }, CONT, CONT, ASCII },
    {  true, { 0xEE, 0xEF }, CONT, CONT },
    { false, { 0xE1, 0xEC }, CONT, ASCII },
    { false, { 0xE1, 0xEC }, CONT, CONT, CONT },

    -- 4-byte sequences
    { false, { 0xF1, 0xF3 }, },
    { false, { 0xF1, 0xF3 }, CONT },
    { false, { 0xF1, 0xF3 }, CONT, CONT },
    {  true, { 0xF1, 0xF3 }, CONT, CONT, CONT },
    { false, { 0xF1, 0xF3 }, CONT, CONT, ASCII },
    {  true, { 0xF1, 0xF3 }, CONT, CONT, CONT, ASCII },
    { false, { 0xF1, 0xF3 }, CONT, CONT, CONT, CONT },

    -- Stray continuation bytes
    { false, CONT, ANY },
    { false, ASCII, CONT },
    { false, ASCII, CONT, CONT },
    { false, ASCII, CONT, CONT, CONT },
    { false, ASCII, CONT, CONT, CONT, CONT },

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

    -- No consecutive leader bytes
    { false, { 0xC0, 0xFF }, { 0xC0, 0xFF }, CONT },
    { false, { 0xC0, 0xFF }, { 0xC0, 0xFF }, CONT, CONT },
    { false, { 0xC0, 0xFF }, { 0xC0, 0xFF }, CONT, CONT, CONT },

    -- Various other cases that probably won't fail, but are here to check that
    -- we at least check every permutation of two bytes in a row.
    { false, ASCII, { 0xC0, 0xC1 }, CONT },
    {  true, ASCII, { 0xC2, 0xDF }, CONT },
    { false, ASCII, { 0xE0, 0xE0 }, { 0x00, 0x9F }, CONT },
    {  true, ASCII, { 0xE0, 0xE0 }, { 0xA0, 0xBF }, CONT },
    {  true, ASCII, { 0xE1, 0xEC }, CONT, CONT },
    {  true, ASCII, { 0xED, 0xED }, { 0x80, 0x9F }, CONT },
    { false, ASCII, { 0xED, 0xED }, { 0xA0, 0xBF }, CONT },
    {  true, ASCII, { 0xEE, 0xEF }, CONT, CONT },
    { false, ASCII, { 0xF0, 0xF0 }, { 0x80, 0x8F }, CONT, CONT },
    {  true, ASCII, { 0xF0, 0xF0 }, { 0x90, 0xBF }, CONT, CONT },
    {  true, ASCII, { 0xF1, 0xF3 }, CONT, CONT, CONT },
    {  true, ASCII, { 0xF4, 0xF4 }, { 0x80, 0x8F }, CONT, CONT },
    { false, ASCII, { 0xF5, 0xFF }, CONT, CONT, CONT },
    { false, { 0xC0, 0xFF }, ASCII },
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
        --local result = expected
        --if count >= skip then
        --    result = validate(str, len)
        --end
        if result ~= expected then
            fails = fails + 1
            print(('failure on test %s, arch %s: got %s, expected %s'):format(
                    count, name, result, expected))
            print(astr(buffer))
            assert(false)
        end
        count = count + 1
    end
    return count, fails
end

-- Keep track of which byte combinations have been seen for the first two bytes
-- We make sure that every combination is tested.
local seen = ffi.new('char [256][256]')

for i = 0, 255 do
    for j = 0, 255 do
        assert(seen[i][j] == 0)
    end
end

local count, fails = 0, 0
for idx, test in ipairs(TEST_CASES) do
    local expected = table.remove(test, 1)
    local lo_1, hi_1 = unpack(table.remove(test, 1))
    -- Loop through various frame shifts, to make sure we catch any issues due
    -- to vector alignment
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
                if not any_valid and (#test > 0 or offset > 0) then
                    break
                end

                seen[buffer[1]][buffer[2]] = true

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

for i = 0, 255 do
    for j = 0, 255 do
        assert(seen[i][j] == 1, ('fail at [%2X][%2X]'):format(i, j))
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
