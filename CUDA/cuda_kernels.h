#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda.h>
#include <stdio.h>
#include <arrayfire.h>
#include "TPCDI_Utils.h"
#ifndef ULL
    #define ULL
typedef unsigned long long ull;
#endif
__global__
static void IsExists(ull *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {
    const ull id = blockIdx.x * blockDim.x + threadIdx.x;

	 ull i = id / set_size;
	 ull j = id % set_size;
	 bool b = id < bag_size * set_size && set[j] == bag[2 * i];
	 if (b) result[k] = 1;
}
void inline bagSetIntersect(af::array &bag, af::array const &set) {
    using namespace af;
    auto const bag_size = bag.row(0).elements();
    auto const set_size = set.elements();
    auto result = constant(0, dim4(1, bag_size + 1), u64);
#ifdef USING_AF
    auto id = range(dim4(1, bag_size * set_size), 1, u64);
    auto i = id / set_size;
    auto j = id % set_size;
    auto b = moddims(set(j), i.dims()) == moddims(bag(0, i),i.dims());
    auto k = b * i + !b * bag_size;
    result(k) = 1;
#else
    ull const threadLimit = 1024;
    ull const threadCount = bag_size * set_size;
    ull const blocks = (threadCount/threadLimit) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threadLimit, 1, 1);

    auto set_ptr = set.device<ull>();
    auto result_ptr = result.device<ull>();
    auto bag_ptr = bag.device<ull>();
    af::sync();

    IsExists<<<grid, block>>>(result_ptr, bag_ptr, set_ptr, bag_size, set_size);

    bag.unlock();
    set.unlock();
    result.unlock();
#endif
    result = result.cols(0, end - 1);
    bag = bag(span, where(result));
    bag.eval();
}

__global__
static void index_scatter(ull const *il, ull const *ir, ull const *cl, ull const *cr, ull const *outpos,
                     ull  *l, ull *r, ull const x, ull const y, ull const z, ull const dump) {

    ull const id = blockIdx.x * blockDim.x + threadIdx.x;

    bool b = id < x * y * z;
    ull i = id / y / z * b;
    ull j = id / z % y;
    ull k = id % z;
    ull left = cl[i];
    ull right = cr[i];
    ull pos = outpos[i];

    b = b && !(j / left) && !(k / right);
    if (b) {
        left = pos + left * k + j;
        right = pos + right * j + k;

        l[left] = il[i] + j;
        r[right] = ir[i] + k;
    }
}

void inline joinScatter(af::array &lhs, af::array &rhs, ull const equals) {
    using namespace af;
    using namespace TPCDI_Utils;
    auto left_count = accum(join(1, constant(1, 1, u64), (diff1(lhs.row(0), 1) > 0).as(u64)), 1) - 1;
    left_count = flipdims(histogram(left_count, left_count.elements())).as(u64);
    left_count = left_count(left_count > 0);
    auto left_max = sum<unsigned int>(max(left_count, 1));
    auto left_idx = scan(left_count, 1, AF_BINARY_ADD, false);

    auto right_count = accum(join(1, constant(1, 1, u64), (diff1(rhs.row(0), 1) > 0).as(u64)), 1) - 1;
    right_count = flipdims(histogram(right_count, right_count.elements())).as(u64);
    right_count = right_count(right_count > 0);
    auto right_max = sum<unsigned int>(max(right_count, 1));
    auto right_idx = scan(right_count, 1, AF_BINARY_ADD, false);

    auto output_pos = right_count * left_count;
    auto output_size = sum<ull>(output_pos);
    output_pos = scan(output_pos, 1, AF_BINARY_ADD, false);
    array left_out(1, output_size + 1, u64);
    array right_out(1, output_size + 1, u64);
#ifdef USING_AF
    auto i = range(dim4(1, equals * left_max * right_max), 1, u64);
    auto j = i / right_max % left_max;
    auto k = i % right_max;
    i = i / left_max / right_max;
    auto b = !(j / left_count(i)) && !(k / right_count(i));
    left_out(b * (output_pos(i) + left_count(i) * k + j) + !b * output_size) = left_idx(i) + j;
    right_out(b * (output_pos(i) + right_count(i) * j + k) + !b * output_size) = right_idx(i) + k;
#else
    auto idx_l = left_idx.device<ull>();
    auto idx_r = right_idx.device<ull>();
    auto count_l = left_count.device<ull>();
    auto count_r = right_count.device<ull>();
    auto pos = output_pos.device<ull>();
    auto left = left_out.device<ull>();
    auto right = right_out.device<ull>();
    af::sync();
    ull const threadLimit = 1024;
    ull const threadCount = equals * left_max * right_max;
    ull const blocks = (threadCount/threadLimit) + 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threadLimit, 1, 1);
    index_scatter<<<grid, block>>>(idx_l, idx_r, count_l, count_r, pos, left, right, equals, left_max, right_max, output_size);

    left_idx.unlock();
    right_idx.unlock();
    left_count.unlock();
    right_count.unlock();
    output_pos.unlock();
    left_out.unlock();
    right_out.unlock();
#endif
    left_out = left_out.cols(0, end - 1);
    right_out = right_out.cols(0, end - 1);
    lhs = lhs(1, left_out);
    rhs = rhs(1, right_out);
    lhs.eval();
    rhs.eval();
}

#endif