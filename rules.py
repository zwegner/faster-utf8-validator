import os

def rules(ctx):
    files = ['z_validate']
    c_flags = ['-fcolor-diagnostics', '-std=gnu11', '-march=native',
            '-Wall', '-Wextra', '-Werror']
    configs = [
        ['avx2/rel', ['-DAVX2', '-O3']],
        ['avx2/deb', ['-DAVX2', '-g']],
        ['sse4/rel', ['-DSSE4', '-O3']],
        ['sse4/deb', ['-DSSE4', '-g']],
    ]

    for [conf_path, conf_flags] in configs:
        o_files = []
        for f in files:
            c_file = '%s.c' % f
            o_file = '_out/%s/%s.o' % (conf_path, f)
            d_file = '_out/%s/%s.d' % (conf_path, f)
            cmd = ['cc', '-o', o_file, '-c', c_file, '-MD', *c_flags,
                    *conf_flags]
            ctx.add_rule(o_file, [c_file], cmd, d_file=d_file)
            o_files.append(o_file)

        # Main shared library
        bin_file = '_out/%s/zval.so' % conf_path
        ctx.add_rule(bin_file, o_files,
            ['cc', '-shared', '-o', bin_file, *c_flags, *o_files])
