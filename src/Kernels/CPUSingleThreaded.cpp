#if (!defined(ARRAYFIRE_TPCDI_SINGLE_THREADED_H) && !defined(USING_OPENCL) && !defined(USING_CUDA))
#define ARRAYFIRE_TPCDI_SINGLE_THREADED_H
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "Kernels.h"

typedef unsigned long long ull;
typedef unsigned long long int ulli;

void launchBagSet(char *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {
    ull start = 0;
    for (int n = 0; n < bag_size; ++n) {
        for (ull i = start; i < set_size; start = ++i) {
            if (set[i] > bag[n]) break;
            if (set[i] != bag[n]) continue;
            result[n] = 1;
            break;
        }
    }
}

void launchHashIntersect(char *result, ull const *bag, ull const *ht_val, ull const *ht_ptr, ull const *ht_occ,
                                unsigned int const buckets, ull const bag_size) {
    for (ull i = 0; i < bag_size; ++i) {
        auto key = bag[i] % buckets;
        auto val = result[i];
        auto len = ht_occ[key];
        auto ptr = ht_ptr[key];
        for (uint j = 0; j < len && !val; ++j) {
            val = bag[i] == ht_val[ptr + j];
        }
        result[i] = val;
    }
}

void inline lauchJoinScatter(ull const *l_idx, ull const *r_idx, ull const *l_cnt, ull const *r_cnt, ull const *outpos, ull *l, ull *r,
                             ull const equals, ull const left_max, ull const right_max, ull const out_size) {
    for (int i = 0; i < equals; ++i) {
        auto jlim = l_cnt[i];
        auto klim = r_cnt[i];
        auto left = l_idx[i];
        auto right = r_idx[i];
        auto pos = outpos[i];
        for (int j = 0; j < jlim; ++j) {
            for (int k = 0; k < klim; ++k) {
                auto idx = pos + jlim * k + j;
                l[idx] = left + j;
                r[idx] = right + k;
            }
        }
    }
}

void launchStringGather(unsigned char *output, ull const *idx, unsigned char const *input, ull const size, ull const rows, ull const loops) {
    for (ull i = 0; i < rows; ++i) {
        auto x = idx[3 * i];
        auto y = idx[3 * i + 1];
        auto z = idx[3 * i + 2];
        for (ull j = 0; j < y; ++j) {
            output[z + j] = input[x + j] * (j != (y - 1));
        }
    }
}

void launchStringComp(bool *output, unsigned char const *left, unsigned char const *right,
                    ull const *l_idx, ull const *r_idx, ull const rows) {
    for (int i = 0; i < rows; ++i) {
        output[i] = !strcmp((char*)(left + l_idx[2 * i]), (char*)(right + r_idx[2 * i]));
    }
}

void launchStringComp(bool *output, unsigned char const *left, unsigned char const *right, ull const *l_idx,  ull const rows, ull const loops) {
    for (int i = 0; i < rows; ++i) {
        output[i] = !strcmp((char*)(left + l_idx[2 * i]), (char*)right);
    }
}
template<typename T> inline T convert(const unsigned char *start);
template<> inline float convert<float>(const unsigned char *start) {
    return std::strtof((char const*)start, nullptr);
}
template<> inline double convert<double>(const unsigned char *start) {
    return std::strtod((char const*)start, nullptr);
}
template<> inline unsigned char convert<unsigned char>(const unsigned char *start) {
    return (unsigned char)std::strtoul((char const*)start, nullptr, 10);
}
template<> inline unsigned short convert<unsigned short>(const unsigned char *start) {
    return (unsigned short)std::strtoul((char const*)start, nullptr, 10);
}
template<> inline unsigned int convert<unsigned int>(const unsigned char *start) {
    return std::strtoul((char const*)start, nullptr, 10);
}
template<> inline ull convert<ull>(const unsigned char *start) {
    return std::strtoull((char const*)start, nullptr, 10);
}
template<> inline short convert<short>(const unsigned char *start) {
    return (short)std::strtol((char const*)start, nullptr, 10);
}
template<> inline int convert<int>(const unsigned char *start) {
    return (int)std::strtol((char const*)start, nullptr, 10);
}
template<> inline long long convert<long long>(const unsigned char *start) {
    return std::strtoll((char const*)start, nullptr, 10);
}

template<typename T> void launchNumericParse(T *output, ull const * idx, unsigned char const *input, ull const rows) {
    for (ull i = 0; i < rows; ++i) {
        auto start = input + idx[2 * i];
        output[i] = *(start) == '\0' ? 0 : convert<T>(start);
    }
}

#endif //ARRAYFIRE_TPCDI_SINGLE_THREADED_H