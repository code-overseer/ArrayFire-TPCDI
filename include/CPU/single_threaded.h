#if (!defined(ARRAYFIRE_TPCDI_SINGLE_THREADED_H) && !defined(USING_OPENCL) && !defined(USING_CUDA))
#define ARRAYFIRE_TPCDI_SINGLE_THREADED_H
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef ULL
    #define ULL
typedef unsigned long long ull;
typedef unsigned long long int ulli;
#endif

void inline launchBagSet(char *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {
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

void inline lauchJoinScatter(ull const *l_idx, ull const *r_idx, ull const *l_cnt, ull const *r_cnt, ull const *outpos, ull *l, ull *r,
                             ull const equals, ull const left_max, ull const right_max, ull const out_size) {
    for (int i = 0; i < equals; ++i) {
        for (int j = 0; j < l_cnt[i]; ++j) {
            for (int k = 0; k < r_cnt[i]; ++k) {
                auto idx = outpos[i] + l_cnt[i] * k + j;
                l[idx] = l_idx[i] + j;
                r[idx] = r_idx[i] + k;
            }
        }
    }
}

void inline launchStringGather(unsigned char *output, ull const *idx, unsigned char const *input, ull const size, ull const rows, ull const loops) {
    for (ull i = 0; i < rows; ++i) {
        for (ull j = 0; j < idx[3 * i + 1]; ++j) {
            output[idx[3 * i + 2] + j] = input[idx[3 * i] + j] * (j != (idx[3 * i + 1] - 1));
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
template<typename T> void inline launchNumericParse(T *output, ull const * idx, unsigned char const *input, ull const rows, ull const loops) {
    for (ull i = 0; i < rows; ++i) {
        auto start = input + idx[2 * i];
        output[i] = *(start) == '\0' ? 0 : convert<T>(start);
    }
}

#endif //ARRAYFIRE_TPCDI_SINGLE_THREADED_H