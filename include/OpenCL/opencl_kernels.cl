__kernel void intersect_kernel(__global char *result, __global ulong const *bag,
        __global ulong const *set, ulong const bag_size, ulong const set_size) {

    ulong const id = get_global_id(0);

    ulong const i = id / set_size;
    ulong const j = id % set_size;
    bool b = id < bag_size * set_size;
    if (b && set[j] == bag[i * b]) result[i] = 1;
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

    if (b && !(j / left) && !(k / right)) {
        left = pos + left * k + j;
        l[left] = il[i] + j;
        r[left] = ir[i] + k;
    }
}

__kernel void string_gather(__global uchar *output, __global ulong const *idx, __global uchar const *input,
ulong const size, ulong const rows, ulong const loops) {

    ulong const id = get_global_id(0);
    ulong const r = id / loops;
    ulong const l = id % loops;
    if (r < rows) {
        ulong const istart = idx[3 * r];
        ulong const len = idx[3 * r + 1];
        ulong const ostart = idx[3 * r + 2];
        bool b = l < len;
        bool c = l != len - 1;
        output[b * (ostart + l) + !b * (size - 1)] = b * input[istart + b * l] * c;
    }
}

__kernel void str_concat(__global uchar *output, __global ulong const *out_idx, __global uchar const *left,
__global ulong const *left_idx,  __global uchar const *right, __global ulong const *right_idx,
ulong const size, ulong const rows, ulong const loops) {
    ulong const id = get_global_id(0);
    ulong const r = id / loops;
    ulong const l = id % loops;
    if (r < rows) {
        ulong const l_start = left_idx[2 * r];
        ulong const r_start = right_idx[2 * r];
        ulong const o_start = out_idx[2 * r];
        ulong const l_len = left_idx[2 * r + 1] - 1;
        ulong const o_len = out_idx[2 * r + 1];

        bool b = l < l_len;
        bool c = (l < o_len) ^ b;
        bool d = l != o_len - 1;

        output[c * (o_start + l) + !c * (size - 1)] = (b * left[l_start + b * l] + c * right[r_start + c * l]) * d;
    }
}