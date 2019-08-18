#ifndef ARRAYFIRE_TPCDI_KERNELINTERFACE_H
#define ARRAYFIRE_TPCDI_KERNELINTERFACE_H

#include <arrayfire.h>
class AFHashTable;

af::array bagSetIntersect(af::array const &bag, af::array const &set);

af::array hashIntersect(af::array const &bag, AFHashTable const &ht);

void joinScatter(af::array &lhs, af::array &rhs, unsigned long long equals);

af::array stringGather(af::array const &input, af::array &indexer);

af::array stringComp(af::array const &lhs, af::array const &rhs, af::array const &l_idx, af::array const &r_idx);

af::array stringComp(af::array const &lhs, char const *rhs, af::array const &l_idx);

template<typename T>
af::array numericParse(af::array const &input, af::array const &indexer);

#endif //ARRAYFIRE_TPCDI_KERNELINTERFACE_H
