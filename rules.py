def rules(ctx):
    cc = 'gcc-8.4'
    files = ['z_validate']
    gen_dir = '_out/gen'
    c_flags = [#'-fcolor-diagnostics',
 '-fPIC', '-std=gnu11', '-march=native', '-fdiagnostics-color=always',
 '-I%s' % gen_dir,
            '-Wall', '-Wextra', '-Werror']

    #c_flags += ['-std=c++11']

#-Wa,-mattr=+avx512vbmi -Wa,-march=cannonlake -march=cannonlake -mavx512vbmi -mbmi2

    configs = [
        ['avx512/rel', ['-DAVX512_VBMI', '-O3']],
        ['avx512/deb', ['-DAVX512_VBMI', '-g']],
        ['avx2/rel', ['-DAVX2', '-O3']],
        ['avx2/deb', ['-DAVX2', '-g']],
        ['sse4/rel', ['-DSSE4', '-O3']],
        ['sse4/deb', ['-DSSE4', '-g']],
        ['neon/rel', ['-DNEON', '-O3']],
        ['neon/deb', ['-DNEON', '-g']],
    ]

    # Generated tables
    table_path = '%s/table.h' % gen_dir
    ctx.add_rule(table_path, ['gen_table.py'],
            ['python3', 'gen_table.py', table_path])

    for [conf_path, conf_flags] in configs:
        o_files = []
        for f in files:
            #c_file = '%s.cpp' % f
            c_file = '%s.c' % f
            o_file = '_out/%s/%s.o' % (conf_path, f)
            d_file = '_out/%s/%s.d' % (conf_path, f)
            cmd = [cc, '-o', o_file, '-c', c_file, '-MD', *c_flags,
                    *conf_flags]
            ctx.add_rule(o_file, [c_file], cmd, d_file=d_file,
                    order_only_deps=[table_path])
            o_files.append(o_file)

        # Main shared library
        bin_file = '_out/%s/zval.so' % conf_path
        ctx.add_rule(bin_file, o_files,
            [cc, '-shared', '-o', bin_file, *c_flags, *o_files])

        # Assembly output
        main_obj = '_out/%s/z_validate.o' % conf_path
        asm_out = '_out/%s/zval.s' % conf_path
        args = '-Mintel' if 'neon' not in conf_path else ''
        ctx.add_rule(asm_out, [main_obj], ['sh', '-c',
            'objdump -d %s %s > %s' % (args, main_obj, asm_out)])
