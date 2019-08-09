//
// Created by Bryan Wong on 2019-06-20.
//

#include "include/BatchFunctions.h"

af::array BatchFunctions::batchEqual(af::array const &rhs, af::array const &lhs) {
    return rhs == lhs;
}

af::array BatchFunctions::batchNotEqual(af::array const &rhs, af::array const &lhs) {
    return rhs != lhs;
}

af::array BatchFunctions::batchLess(af::array const &rhs, af::array const &lhs) {
    return rhs < lhs;
}
af::array BatchFunctions::batchGreater(af::array const &rhs, af::array const &lhs) {
    return rhs > lhs;
}
af::array BatchFunctions::batchGE(af::array const &rhs, af::array const &lhs) {
    return rhs >= lhs;
}
af::array BatchFunctions::batchLE(af::array const &rhs, af::array const &lhs) {
    return rhs <= lhs;
}
af::array BatchFunctions::batchAdd(af::array const &rhs, af::array const &lhs) {
    return rhs + lhs;
}
af::array BatchFunctions::batchMult(af::array const &rhs, af::array const &lhs) {
    return rhs * lhs;
}
af::array BatchFunctions::batchSub(af::array const &lhs, af::array const &rhs) {
    return lhs - rhs;
}

af::array BatchFunctions::batchDiv(af::array const &rhs, af::array const &lhs) {
    return rhs / lhs;
}

af::array BatchFunctions::batchMod(af::array const &rhs, af::array const &lhs) {
    return rhs % lhs;
}

af::array BatchFunctions::bitShiftLeft(af::array const &lhs, af::array const &rhs) {
    return lhs << rhs;
}

af::array BatchFunctions::bitShiftRight(af::array const &lhs, af::array const &rhs) {
    return lhs >> rhs;
}

af::array BatchFunctions::exOr(af::array const &lhs, af::array const &rhs) {
    return lhs ^ rhs;
}

af::array BatchFunctions::bitAnd(af::array const &lhs, af::array const &rhs) {
    return lhs & rhs;
}

af::array BatchFunctions::bitOr(af::array const &lhs, af::array const &rhs) {
    return lhs | rhs;
}

af::array BatchFunctions::batchStrCmp(af::array const &rhs, af::array const &lhs) {
    return allTrue(rhs == lhs);
}
