//
// Created by Bryan Wong on 2019-06-20.
//
#ifndef ARRAYFIRE_TPCDI_BATCHFUNCTIONS_H
#define ARRAYFIRE_TPCDI_BATCHFUNCTIONS_H
#include <arrayfire.h>

namespace BatchFunctions {

    af::array batchNotEqual(af::array const &rhs, af::array const &lhs);
    af::array batchEqual(af::array const &rhs, af::array const &lhs);
    af::array batchLess(af::array const &rhs, af::array const &lhs);
    af::array batchGreater(af::array const &rhs, af::array const &lhs);
    af::array batchGE(af::array const &rhs, af::array const &lhs);
    af::array batchLE(af::array const &rhs, af::array const &lhs);
    af::array batchAdd(af::array const &rhs, af::array const &lhs);
    af::array batchMult(af::array const &rhs, af::array const &lhs);
    af::array batchDiv(af::array const &rhs, af::array const &lhs);
    af::array batchMod(af::array const &rhs, af::array const &lhs);
    af::array batchSub(af::array const &lhs, af::array const &rhs);
    af::array bitShiftLeft(af::array const &lhs, af::array const &rhs);
    af::array bitShiftRight(af::array const &lhs, af::array const &rhs);
    af::array exOr(af::array const &lhs, af::array const &rhs);
    af::array bitAnd(af::array const &lhs, af::array const &rhs);
    af::array bitOr(af::array const &lhs, af::array const &rhs);

}

#endif //ARRAYFIRE_TPCDI_BATCHFUNCTIONS_H
