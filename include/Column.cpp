//
// Created by Bryan Wong on 2019-08-05.
//

#include "include/Column.h"
#include "include/TPCDI_Utils.h"
#include <exception>

af::array Column::hash(bool const sortable) const {
    using namespace TPCDI_Utils;
    if (_type == STRING) return sortable ? byteHash(_device) : polyHash(byteHash(_device));
    if (_type == DATE) return dateHash(_device).as(u64);
    if (_type == TIME) return timeHash(_device).as(u64);
    if (_type == DATETIME) return datetimeHash(_device).as(u64);
    return af::array(_device).as(u64);
}
#define ASSIGN(OP) \
af::array Column::operator==(af::array OP other) { \
    af::array lhs; \
    if (_type == STRING || _type == DATE || _type == TIME || _type == DATETIME) lhs = hash(false); \
    return lhs == other; \
}
    ASSIGN( const &)
    ASSIGN(&&)
#undef ASSIGN

#define ASSIGN(OP) \
af::array Column::operator OP(af::array const &other) { \
    af::array lhs; \
    if (_type == DATE || _type == TIME || _type == DATETIME) lhs = hash(false); \
    if (_type == STRING) { throw std::runtime_error("Invalid column type"); } \
    return lhs OP other; \
} \
af::array Column::operator OP(af::array &&other) { \
    af::array lhs; \
    if (_type == DATE || _type == TIME || _type == DATETIME) lhs = hash(false); \
    if (_type == STRING) { throw std::runtime_error("Invalid column type"); } \
    return lhs OP other; \
}
    ASSIGN(<)
    ASSIGN(>)
    ASSIGN(<=)
    ASSIGN(>=)
#undef ASSIGN

af::array Column::operator==(Column const &other) {
    af::array lhs;
    af::array rhs;
    if (_type != other._type) { throw std::runtime_error("Mismatch column type"); }
    if (_type == STRING || _type == DATE || _type == TIME || _type == DATETIME) {
        lhs = hash(false);
        rhs = other.hash(false);
    }
    if (!lhs.isempty() && !rhs.isempty()) return lhs == rhs;
    return _device == other._device;
}

af::array Column::operator!=(Column const &other) { return !(*this == other); }

af::array::array_proxy &Column::operator()(af::index const &x, af::index const &y) {
    static auto zero = af::constant(0, 1, b8);
    static af::array::array_proxy tmp = zero(0);
    tmp = _device(x, y);
    return  tmp;
}

void Column::refresh() {
    if (_type != STRING) return;
    //TODO string refresh
}

