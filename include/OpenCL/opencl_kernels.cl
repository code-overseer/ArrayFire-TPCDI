__kernel void is_exist_kernel(__global char *result, __global ulong const *bag,
        __global ulong const *set, ulong const bag_size, ulong const set_size) {

    ulong const id = get_global_id(0);

    ulong const i = id / set_size;
    ulong const j = id % set_size;

    if (id < bag_size * set_size && set[j] == bag[2 * i]) result[i] = 1;
}

__kernel void join_scatter(__global ulong const *il, __global ulong const *ir, __global ulong const *cl,
        __global ulong const *cr, __global ulong const *outpos,  __global ulong *l, __global ulong *r,
        ulong const equals, ulong const left_max, ulong const right_max, ulong const out_size) {

    ulong const id = get_global_id(0);
    bool b = id < equals * left_max * right_max;
    ulong const i = id / left_max / right_max * b;
    ulong const j = id / right_max % left_max;
    ulong const k = id % right_max;
    ulong left = cl[i];
    ulong right = cr[i];
    ulong pos = outpos[i];
    b = b && !(j / left) && !(k / right);

    if (b) {
        left = pos + left * k + j;
        l[left] = il[i] + j;
        r[left] = ir[i] + k;
    }
}