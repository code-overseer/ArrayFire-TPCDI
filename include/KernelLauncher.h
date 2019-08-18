#ifndef ARRAYFIRE_TPCDI_KERNELLAUNCHER_H
#define ARRAYFIRE_TPCDI_KERNELLAUNCHER_H

void launchBagSet(char *result, unsigned long long const *bag, unsigned long long const *set,
        unsigned long long bag_size, unsigned long long set_size);

void launchHashIntersect(char *result, unsigned long long const *bag, unsigned long long const *ht_val,
        unsigned long long const *ht_ptr, unsigned long long const *ht_occ, unsigned int buckets, unsigned long long bag_size);

void lauchJoinScatter(unsigned long long const *l_idx, unsigned long long const *r_idx, unsigned long long const *l_cnt,
        unsigned long long const *r_cnt, unsigned long long const *outpos, unsigned long long *l, unsigned long long *r,
        unsigned long long equals, unsigned long long left_max, unsigned long long right_max, unsigned long long out_size);

void launchStringGather(unsigned char *output, unsigned long long const *idx, unsigned char const *input,
        unsigned long long output_size, unsigned long long rows, unsigned long long loops);

void launchStringComp(bool *output, unsigned char const *left, unsigned char const *right,
        unsigned long long const *l_idx, unsigned long long const *r_idx, unsigned long long rows);

template<typename T>
void launchNumericParse(T *output, unsigned long long const * idx, unsigned char const *input,
        unsigned long long rows);

void launchStringComp(bool *output, unsigned char const *left, unsigned char const *right,
        unsigned long long const *l_idx, unsigned long long rows, unsigned long long loops);

#endif //ARRAYFIRE_TPCDI_KERNELLAUNCHER_H
