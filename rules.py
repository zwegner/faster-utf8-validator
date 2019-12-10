import itertools

def rules(ctx):
    cc = ctx.vars.get('cc', 'cc')
    objdump = ctx.vars.get('objdump', 'objdump')
    arch = ctx.vars.get('arch', 'native')
    files = ['z_validate']
    gen_dir = '_out/gen'
    c_flags = ['-I%s' % gen_dir, '-fPIC', '-std=c11', '-march=%s' % arch,
        '-fdiagnostics-color=always', '-Wall', '-Wextra', '-Werror']

    # Create all arch/config combinations
    configs = itertools.product(
        ['avx512_vbmi', 'avx2', 'sse4', 'neon'],
        [['rel', ['-O3']], ['deb', ['-g']]])

    # Generated tables
    table_path = '%s/table.h' % gen_dir
    ctx.add_rule(table_path, ['gen_table.py'],
            ['python3', 'gen_table.py', table_path])

    for [arch, [conf, conf_flags]] in configs:
        flags = ['-D%s' % arch.upper(), *conf_flags, *c_flags]
        out_dir = '_out/%s/%s' % (arch, conf)

        # Object files
        o_files = []
        for f in files:
            c_file = '%s.c' % f
            o_file = '%s/%s.o' % (out_dir, f)
            d_file = '%s/%s.d' % (out_dir, f)
            cmd = [cc, '-o', o_file, '-c', c_file, '-MD', *flags]
            ctx.add_rule(o_file, [c_file], cmd, d_file=d_file,
                    order_only_deps=[table_path])
            o_files.append(o_file)

        # Main shared library
        bin_file = '%s/zval.so' % out_dir
        ctx.add_rule(bin_file, o_files,
            [cc, '-shared', '-o', bin_file, *c_flags, *o_files])

        # Assembly output
        main_obj = '%s/z_validate.o' % out_dir
        asm_out = '%s/zval.s' % out_dir
        args = '-Mintel' if arch != 'neon' else ''
        # Spawn a shell for redirection, since objdump apparently doesn't have -o
        ctx.add_rule(asm_out, [main_obj], ['sh', '-c',
            '"%s" -d %s "%s" > "%s"' % (objdump, args, main_obj, asm_out)])
