//
// Created by Bryan Wong on 2019-06-20.
//

#include "BatchFunctions.h"

af::array BatchFunctions::batchEqual(af::array const &lhs, af::array const &rhs) {
    return lhs == rhs;
}

af::array BatchFunctions::batchLess(af::array const &lhs, af::array const &rhs) {
    return lhs < rhs;
}
af::array BatchFunctions::batchGreater(af::array const &lhs, af::array const &rhs) {
    return lhs > rhs;
}
af::array BatchFunctions::batchGE(af::array const &lhs, af::array const &rhs) {
    return lhs >= rhs;
}
af::array BatchFunctions::batchLE(af::array const &lhs, af::array const &rhs) {
    return lhs <= rhs;
}
af::array BatchFunctions::batchAdd(af::array const &lhs, af::array const &rhs) {
    return lhs + rhs;
}
af::array BatchFunctions::batchMul(af::array const &lhs, af::array const &rhs) {
    return lhs * rhs;
}
af::array BatchFunctions::batchSub(af::array const &lhs, af::array const &rhs) {
    return lhs - rhs;
}

af::array BatchFunctions::batchDiv(af::array const &lhs, af::array const &rhs) {
    return lhs / rhs;
}