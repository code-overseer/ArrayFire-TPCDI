__kernel void intersect_kernel(__global char *result, __global ulong const *bag,
                               __global ulong const *set, ulong const bag_size, ulong const set_size,
                               ulong const offset) {

    ulong id = get_global_id(0);
    id += offset;

    ulong const i = id / set_size;
    ulong const j = id % set_size;
    bool b = id - offset < bag_size * set_size;
    if (b && set[j] == bag[i * b]) result[i] = 1;

}

__kernel void hash_intersect(__global char *result, __global ulong const *bag, __global ulong const *ht_val,
                            __global ulong const *ht_ptr, __global ulong const *ht_occ, uint const buckets, ulong const bag_size) {
    ulong id = get_global_id(0);
    if (id < bag_size) {
        ulong const val = bag[id];
        uint const key = val % buckets;
        uint const len = ht_occ[key];
        ulong const ptr = ht_ptr[key];

        char out = 0;
        for (uint i = 0; i < len; ++i) {
            out |= (ht_val[ptr + i] == val);
        }
        result[id] = out;
    }
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
        pos += (left * k + j);
        l[pos] = il[i] + j;
        r[pos] = ir[i] + k;
    }
}

__kernel void string_gather(__global uchar *output, __global ulong const *idx, __global uchar const *input,
ulong const size, ulong const rows, ulong const loops) {

    ulong const id = get_global_id(0);
    ulong const r = id / loops;
    ulong const l = id % loops;
    if (r < rows) {
        ulong const istart = idx[3 * r];
        ulong const len = idx[3 * r + 1] - 1;
        ulong const ostart = idx[3 * r + 2];
        bool const b = l < len;
        ulong const k = b * l + !b * len;
        output[ostart + k] = b * input[istart + k];
    }
}

__kernel void str_cmp(__global bool *output, __global uchar const *left, __global uchar const *right,
__global ulong const *l_idx, __global ulong const *r_idx, ulong const rows) {
    ulong const id = get_global_id(0);
    if (id < rows) {
        ulong const l_start = l_idx[2 * id];
        ulong const len = l_idx[2 * id + 1];
        ulong const r_start = r_idx[2 * id];
        bool out = output[id];
        for (ulong i = 0; i < len; ++i) {
            out &= left[l_start + i] == right[r_start + i];
        }
        output[id] = out;
    }
}

__kernel void str_concat(__global uchar *output, __global ulong const *out_idx, __global uchar const *left,
__global ulong const *left_idx,  __global uchar const *right, __global ulong const *right_idx,
ulong const size, ulong const rows, ulong const loops) {
    ulong i = get_global_id(0);
    ulong j = i / loops;
    ulong k = i % loops;
    if (j < rows) {
        ulong const l_start = left_idx[2 * j];
        ulong const r_start = right_idx[2 * j];
        ulong const o_start = out_idx[2 * j];
        ulong const l_end = left_idx[2 * j + 1] - 1;
        ulong const o_end = out_idx[2 * j + 1] - 1;
        ulong const r_end = right_idx[2 * j + 1] - 1;

        bool const b = k < o_end;
        bool const c = k < l_end;
        bool const d = c ^ b;
        i = b * k + !b * o_end;
        j = c * k + !c * l_end;
        k = d * (k - l_end) + !b * r_end;
        output[o_start + i] = c * left[l_start + j] + d * right[r_start + k];
    }
}

#define PARSER_FUNC(TYPE) \
__kernel void parser_##TYPE (__global TYPE *output, __global ulong const *idx, __global uchar const *input, \
    ulong const row_num) { \
    ulong const id = get_global_id(0); \
    if (id < row_num) { \
        long const s = idx[2 * id]; \
        long const len = idx[2 * id + 1] - 1; \
        TYPE number = 0; \
        uchar dec = 0; \
        bool frac = 0; \
        bool neg = input[s] == '-'; \
        for (long i = 0; i < len; ++i) { \
            uchar digit = input[s + i]; \
            bool b = digit >= '0' && digit <= '9'; \
            frac |= digit == '.'; \
            dec += b && frac; \
            bool c = !dec && b; \
            number = number * (c * 10 + !c) + b * (digit - '0') / (TYPE)pown(10.0, dec); \
        } \
        output[id] = number * (!neg - neg); \
    } \
}

PARSER_FUNC(bool)
PARSER_FUNC(uchar)
PARSER_FUNC(float)
PARSER_FUNC(double)
PARSER_FUNC(ushort)
PARSER_FUNC(short)
PARSER_FUNC(uint)
PARSER_FUNC(int)
PARSER_FUNC(ulong)
PARSER_FUNC(long)

#undef PARSER_FUNC