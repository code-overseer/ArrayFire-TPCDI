//
// Created by Bryan Wong on 2019-07-31.
//

#ifndef ARRAYFIRE_TPCDI_VECTOR_FUNCTIONS_H
#define ARRAYFIRE_TPCDI_VECTOR_FUNCTIONS_H
#include <thread>
#include <vector>
#include <cstdio>
#include <algorithm>
#ifndef ULL
    #define ULL
typedef unsigned long long ull;
typedef unsigned long long int ulli;
#endif

static void isExist(ull *result, ull const *input, ull const *set, ull *start, ull const set_size, ull const j) {
    for (ull i = start[0]; i < set_size; start[j] = ++i) {
        if (set[i] > *input) return;
        if (set[i] ^ *input) continue;
        *result = 1;
        return;
    }
}

static void isExist_single(ull &result, ull const &input, ull const *set,  ulli &start, ull const set_size) {
    for (ull i = start; i < set_size; start = ++i) {
        if (set[i] > input) return;
        if (set[i] ^ input) continue;
        result = 1;
        return;
    }
}

void inline launch_IsExist(ull *result, ull const *input, ull const *comparison, ull const i_size, ull const comp_size) {
    if (i_size * comp_size > 10000000llU) {
        ull const limit = std::thread::hardware_concurrency() - 1;
        ull const threadCount = i_size;
        std::vector<ull> s((size_t)limit);
        for (int i = 0; i < limit; ++i) s[i] = 0;

        std::thread threads[limit];
        for (ull i = 0; i < threadCount / limit + 1; ++i) {
            auto jlim = (i == threadCount / limit) ? threadCount % limit : limit;
            for (ull j = 0; j < jlim; ++j) {
                auto n = i * limit + j;
                threads[j] = std::thread(isExist, result + n, input + 2 * n, comparison, s.data(), comp_size, j);
            }
            for (ull j = 0; j < jlim; ++j) threads[j].join();
            for (ull j = 0; j < jlim; ++j) s[0] = (s[0] < s[j]) ? s[j] : s[0];
        }
    } else {
        ull s = 0;
        for (int n = 0; n < i_size; ++n) isExist_single(result[n], input[2 * n], comparison, s, comp_size);
    }
}

#endif //ARRAYFIRE_TPCDI_VECTOR_FUNCTIONS_H
