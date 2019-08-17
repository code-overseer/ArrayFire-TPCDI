__kernel void str_cmp_single(__global bool *output, __global uchar const *left, __global uchar const *right,
__global ulong const *l_idx, ulong const rows) {

    ulong const id = get_global_id(0);
    if (id < rows) {
        ulong const l_start = l_idx[2 * id];
        bool out = output[id];
        #pragma unroll LOOP_LENGTH
        for (ulong i = 0; i < LOOP_LENGTH; ++i) {
            out &= left[l_start + i] == right[i];
        }
        output[id] = out;
    }
}