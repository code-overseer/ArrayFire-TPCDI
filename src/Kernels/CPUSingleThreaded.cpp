#if !defined(USING_CUDA) && !defined(USING_OPENCL)
#include "Kernels.h"
#include <cstdlib>
#include <cstring>

typedef unsigned long long ull;

void launchCrossIntersect(char *result, unsigned long long const *bag, unsigned long long const *set,
                          unsigned long long bag_size, unsigned long long set_size) {
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

void launchHashIntersect(char *result, unsigned long long const *bag, unsigned long long const *ht_val,
                         unsigned long long const *ht_ptr, unsigned long long const *ht_occ, unsigned int buckets, unsigned long long bag_size) {
    for (ull i = 0; i < bag_size; ++i) {
        auto key = bag[i] % buckets;
        auto val = result[i];
        auto len = ht_occ[key];
        auto ptr = ht_ptr[key];
        for (int j = 0; j < len && !val; ++j) {
            val = bag[i] == ht_val[ptr + j];
        }
        result[i] = val;
    }
}

void lauchJoinScatter(unsigned long long const *l_idx, unsigned long long const *r_idx, unsigned long long const *l_cnt,
                      unsigned long long const *r_cnt, unsigned long long const *outpos, unsigned long long *l, unsigned long long *r,
                      unsigned long long equals, unsigned long long left_max, unsigned long long right_max, unsigned long long out_size) {
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

void launchStringGather(unsigned char *output, unsigned long long const *idx, unsigned char const *input,
                        unsigned long long output_size, unsigned long long rows, unsigned long long loops) {
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
                      unsigned long long const *l_idx, unsigned long long const *r_idx, unsigned int const *mask, unsigned long long rows) {
    for (int j = 0; j < rows; ++j) {
        unsigned int i = mask[j];
        output[i] = !strcmp((char*)(left + l_idx[2 * i]), (char*)(right + r_idx[2 * i]));
    }
}

void launchStringComp(bool *output, unsigned char const *left, unsigned char const *right,
                      unsigned long long const *l_idx, unsigned long long rows, unsigned long long loops) {
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

template<typename T>
void launchNumericParse(T *output, unsigned long long const * idx, unsigned char const *input,
                        unsigned long long rows) {
    for (ull i = 0; i < rows; ++i) {
        auto start = input + idx[2 * i];
        output[i] = *(start) == '\0' ? 0 : convert<T>(start);
    }
}

#define PARSER(TYPE) \
template void launchNumericParse<TYPE>(TYPE *output, ull const * idx, unsigned char const *input, ull const rows);

PARSER(unsigned char)
PARSER(float)
PARSER(double)
PARSER(unsigned short)
PARSER(short)
PARSER(unsigned int)
PARSER(int)
PARSER(ull)
PARSER(long long)

#undef PARSER

#endif