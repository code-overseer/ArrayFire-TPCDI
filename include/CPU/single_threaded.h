#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-err34-c"
#if (!defined(ARRAYFIRE_TPCDI_SINGLE_THREADED_H) && !defined(USING_OPENCL) && !defined(USING_CUDA))
#define ARRAYFIRE_TPCDI_SINGLE_THREADED_H
#include <thread>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <arrayfire.h>
#include <cstdlib>
#include "include/TPCDI_Utils.h"
#include "include/BatchFunctions.h"
#ifndef ULL
    #define ULL
typedef unsigned long long ull;
typedef unsigned long long int ulli;
#endif

void inline launchBagSet(char *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {
    ull start = 0;
    for (int n = 0; n < bag_size; ++n) {
        for (ull i = start; i < set_size; start = ++i) {
            if (set[i] > bag[2 * n]) return;
            if (set[i] ^ bag[2 * n]) continue;
            result[n] = 1;
            return;
        }
    }
}

void inline lauchJoinScatter(ull const *il, ull const *ir, ull const *cl, ull const *cr, ull const *outpos,  ull *l, ull *r,
    ull const equals, ull const left_max, ull const right_max, ull const out_size) {

    for (int i = 0; i < equals; ++i) {
        for (int j = 0; j < cl[i]; ++j) {
            for (int k = 0; k < cr[i]; ++k) {
                auto idx = outpos[i] + cl[i] * k + j;
                l[idx] = il[i] + j;
                r[idx] = ir[i] + k;
            }
        }
    }
}

void inline launchStringGather(unsigned char *output, ull const *idx, unsigned char const *input, ull const size, ull const rows, ull const loops) {
    for (ull i = 0; i < rows; ++i) {
        for (ull j = 0; j < idx[3 * i + 1]; ++j) {
            output[idx[3 * i + 2] + j] = input[idx[3 * i] + j];
        }
    }
}

void inline launchStringComp(bool *output, unsigned char const *left, unsigned char const *right,
                    ull const *l_idx, ull const *r_idx, ull const rows, ull const loops) {
    for (int i = 0; i < rows; ++i) {
        output[i] = !strcmp((char*)(left + l_idx[2 * i]), (char*)(right + r_idx[2 * i]));
    }
}

void inline launchStringComp(bool *output, unsigned char const *left, unsigned char const *right, ull const *l_idx,  ull const rows, ull const loops) {
    for (int i = 0; i < rows; ++i) {
        output[i] = !strcmp((char*)(left + l_idx[2 * i]), (char*)right);
    }
}

template<typename T> inline T convert(unsigned char *start);
template<> inline float convert<float>(unsigned char *start) {
    return std::strtof((char*)start, nullptr);
}
template<> inline double convert<double>(unsigned char *start) {
    return std::strtod((char*)start, nullptr);
}
template<> inline unsigned char convert<unsigned char>(unsigned char *start) {
    return (unsigned char)std::strtoul((char*)start, nullptr, 0);
}
template<> inline unsigned short convert<unsigned short>(unsigned char *start) {
    return (unsigned short)std::strtoul((char*)start, nullptr, 0);
}
template<> inline short convert<short>(unsigned char *start) {
    return (short)std::strtol((char*)start, nullptr, 0);
}
template<> inline unsigned int convert<unsigned int>(unsigned char *start) {
    return std::strtoul((char*)start, nullptr, 0);
}
template<> inline int convert<int>(unsigned char *start) {
    return (int)std::strtol((char*)start, nullptr, 0);
}
template<> inline ull convert<ull>(unsigned char *start) {
    return std::strtoull((char*)start, nullptr, 0);
}
template<> inline long long convert<long long>(unsigned char *start) {
    return std::strtoll((char*)start, nullptr, 0);
}
template<typename T> void inline launchNumericParse(T *output, ull const * idx, unsigned char const *input, ull const rows, ull const loops) {
    for (ull i = 0; i < rows; ++i) {
        auto start = input + idx[2 * i];
        output[i] = *(start) == '\0' ? 0 : convert<T>(start);
    }
}
//
//void inline bagSetIntersect(af::array &bag, af::array const &set) {
//    using namespace af;
//    auto const bag_size = bag.row(0).elements();
//    auto const set_size = set.elements();
//    auto result = constant(0, dim4(1, bag_size + 1), b8);
//#ifdef USING_AF
//    auto id = range(dim4(1, bag_size * set_size), 1, u64);
//    auto i = id / set_size;
//    auto j = id % set_size;
//    auto b = moddims(set(j), i.dims()) == moddims(bag(0, i), i.dims());
//    auto k = b * i + !b * bag_size;
//    result(k) = 1;
//#else
//    auto result_ptr = result.device<char>();
//    auto set_ptr = set.device<ull>();
//    auto bag_ptr = bag.device<ull>();
//    af::sync();
//
//    launchBagSet(result_ptr, bag_ptr, set_ptr, bag_size, set_size);
//
//    bag.unlock();
//    set.unlock();
//    result.unlock();
//#endif
//    result = result.cols(0, end - 1);
//    bag = bag(span, where(result));
//    bag.eval();
//}
//
//void inline joinScatter(af::array &lhs, af::array &rhs, ull const equals) {
//    using namespace af;
//    using namespace TPCDI_Utils;
//    auto left_count = accum(join(1, constant(1, 1, u64), (diff1(lhs.row(0), 1) > 0).as(u64)), 1) - 1;
//    left_count = hflat(histogram(left_count, left_count.elements())).as(u64);
//    left_count = left_count(left_count > 0);
//    auto left_max = sum<unsigned int>(max(left_count, 1));
//    auto left_idx = (left_count.elements() == 1) ? constant(0, 1, left_count.type())
//                                                 : scan(left_count, 1, AF_BINARY_ADD, false);
//
//    auto right_count = accum(join(1, constant(1, 1, u64), (diff1(rhs.row(0), 1) > 0).as(u64)), 1) - 1;
//    right_count = hflat(histogram(right_count, right_count.elements())).as(u64);
//    right_count = right_count(right_count > 0);
//    auto right_max = sum<unsigned int>(max(right_count, 1));
//    auto right_idx = (right_count.elements() == 1) ? constant(0, 1, right_count.type())
//                                                   : scan(right_count, 1, AF_BINARY_ADD, false);
//
//    auto output_pos = right_count * left_count;
//    auto output_size = sum<ull>(output_pos);
//    output_pos = (output_pos.elements() == 1) ? constant(0, 1, output_pos.type())
//                                              : scan(output_pos, 1, AF_BINARY_ADD, false);
//#ifdef USING_AF
//    array left_out(1, output_size + 1, u64);
//    array right_out(1, output_size + 1, u64);
//    auto i = range(dim4(1, equals * left_max * right_max), 1, u64);
//    auto j = i / right_max % left_max;
//    auto k = i % right_max;
//    i = i / left_max / right_max;
//    auto b = !(j / left_count(i)) && !(k / right_count(i));
//    left_out(b * (output_pos(i) + left_count(i) * k + j) + !b * output_size) = left_idx(i) + j;
//    right_out(b * (output_pos(i) + right_count(i) * j + k) + !b * output_size) = right_idx(i) + k;
//    left_out = left_out.cols(0, end - 1);
//    right_out = right_out.cols(0, end - 1);
//#else
//    array left_out(1, output_size, u64);
//    array right_out(1, output_size, u64);
//    auto idx_l = left_idx.device<ull>();
//    auto idx_r = right_idx.device<ull>();
//    auto count_l = left_count.device<ull>();
//    auto count_r = right_count.device<ull>();
//    auto pos = output_pos.device<ull>();
//    auto left = left_out.device<ull>();
//    auto right = right_out.device<ull>();
//    af::sync();
//
//    lauchJoinScatter(idx_l, idx_r, count_l, count_r, pos, left, right, equals, left_max, right_max, output_size);
//
//    left_idx.unlock();
//    right_idx.unlock();
//    left_count.unlock();
//    right_count.unlock();
//    output_pos.unlock();
//    left_out.unlock();
//    right_out.unlock();
//#endif
//    lhs = lhs(1, left_out);
//    rhs = rhs(1, right_out);
//    lhs.eval();
//    rhs.eval();
//}
//
//af::array inline stringGather(af::array const &input, af::array &indexer) {
//    using namespace af;
//    indexer = join(0, indexer, scan(indexer.row(1), 1, AF_BINARY_ADD, false));
//    indexer.eval();
//    auto const out_size = sum<ull>(indexer.row(1));
//    auto const loops = sum<ull>(max(indexer.row(1), 1));
//    auto const rows = indexer.elements() / 3;
//    auto output = array(out_size, u8);
//
//    #ifdef USING_AF
//    for (ull i = 0; i < loop_length; ++i) {
//        auto b = indexer.row(1) > i;
//        auto c = indexer.row(1) - 1 != i;
//        output(indexer(2, b) + i) = input(indexer(0, b) + i) * c;
//    }
//    output.eval();
//    #else
//    auto out_ptr = output.device<unsigned char>();
//    auto in_ptr = input.device<unsigned char>();
//    auto idx_ptr = indexer.device<ull>();
//    af::sync();
//
//    launchStringGather(out_ptr, idx_ptr, in_ptr, out_size, rows, loops);
//
//    output.unlock();
//    input.unlock();
//    indexer.unlock();
//    #endif
//    indexer.row(0) = (array)indexer.row(2);
//    indexer = indexer.rows(0, 1);
//    indexer.eval();
//    return output;
//}
//
//template<typename T> af::array inline numericParse(af::array const &input, af::array const &indexer) {
//    using namespace af;
//
//    auto const loops = sum<ull>(max(indexer.row(1), 1));
//    auto const rows = indexer.elements() / 2;
//    auto output = array(1, rows + 1, GetAFType<T>().af_type);
//    auto out_ptr = output.device<T>();
//    auto idx_ptr = indexer.device<ull>();
//    auto in_ptr = input.device<unsigned char>();
//    af::sync();
//    launchNumericParse<T>(out_ptr, idx_ptr, in_ptr, rows, loops);
//    output.unlock();
//    input.unlock();
//    indexer.unlock();
//    output = output.cols(0, end - 1);
//    output.eval();
//
//    af::sync();
//
//    return output;
//}
//
//af::array inline stringComp(af::array const &lhs, af::array const &rhs, af::array const &l_idx, af::array const &r_idx) {
//    using namespace af;
//    auto out = l_idx.row(1) == r_idx.row(1);
//    auto loops = sum<ull>(max(l_idx(1, out)));
//
//    auto out_ptr = (bool*)out.device<char>();
//    auto left_ptr = lhs.device<unsigned char>();
//    auto right_ptr = rhs.device<unsigned char>();
//    auto l_idx_ptr = l_idx.device<ull>();
//    auto r_idx_ptr = r_idx.device<ull>();
//    auto const rows = l_idx.elements() / 2;
//    af::sync();
//
//    launchStringComp(out_ptr, left_ptr, right_ptr, l_idx_ptr, r_idx_ptr, rows, loops);
//
//    out.unlock();
//    lhs.unlock();
//    rhs.unlock();
//    l_idx.unlock();
//    r_idx.unlock();
//    out.eval();
//
//    return out;
//}
//
//af::array inline stringComp(af::array const &lhs, char const *rhs, af::array const &l_idx) {
//    using namespace af;
//    auto loops = strlen(rhs) + 1;
//    auto out = l_idx.row(1) == loops;
//
//    auto right = array(loops, rhs).as(u8);
//    auto out_ptr = (bool*)out.device<char>();
//    auto left_ptr = lhs.device<unsigned char>();
//    auto right_ptr = right.device<unsigned char>();
//    auto l_idx_ptr = l_idx.device<ull>();
//    auto const rows = l_idx.elements() / 2;
//    af::sync();
//
//    launchStringComp(out_ptr, left_ptr, right_ptr, l_idx_ptr, rows, loops);
//
//    out.unlock();
//    lhs.unlock();
//    l_idx.unlock();
//    right.unlock();
//    out.eval();
//
//    return out;
//}
#endif //ARRAYFIRE_TPCDI_SINGLE_THREADED_H

#pragma clang diagnostic pop