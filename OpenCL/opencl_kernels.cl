__kernel void is_exist_kernel(__global ulong *result, __global ulong const *bag,
        __global ulong const *set, ulong const bag_size, ulong const set_size) {

    const ulong id = get_global_id(0);

    ulong i = id / set_size;
    ulong j = id % set_size;
    bool b = id < bag_size * set_size && set[j] == bag[2 * i];
    ulong k = b * i + !b * bag_size;

    result[k] = 1;
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
    left = b * (pos + left * k + j) + !b * out_size;
    right = b * (pos + right * j + k) + !b * out_size;

    l[left] = il[i] + j;
    r[right] = ir[i] + k;
}