//

#include "include/Column.h"
#include "include/TPCDI_Utils.h"
#include "include/BatchFunctions.h"
#include <exception>


Column::Column(Column::Proxy &&data, Column::Proxy &&idx, DataType const type) : _type(type) {
    _device = data; _idx = idx;
}

Column::Column(Column::Proxy const &data, Column::Proxy const &idx, DataType const type) : _type(type) {
    _device = data; _idx = idx;
}

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

void Column::flush() {
    if (_type != STRING) return;
    //TODO string refresh
}

Column &Column::operator=(Column &&other) noexcept {
    _device = std::move(other._device);
    _idx = std::move(other._idx);
    _type = other._type;
    _host = other._host;
    _host_idx = other._host_idx;
    other._host = nullptr;
    other._host_idx = nullptr;
    return *this;
}

Column Column::concatenate(Column &bottom) {
    using namespace BatchFunctions;
    flush();
    bottom.flush();
    if (_type != bottom._type) throw std::runtime_error("Type mismatch");
    if (_type == STRING) {
        auto i = bottom._idx;
        i.row(0) = af::batchFunc(i.row(0), af::sum(_idx.col(af::end), 0), batchAdd);
        return Column(join(0, _device, bottom._device), std::move(i));
    }
    return Column(join(0, _device, bottom._device), _type);
}

void Column::toHost(bool const clear) {
    if (_device.bytes()) {
        auto tmp = malloc(_device.bytes());
        _device.host(tmp);
        _host = tmp;
    }
    if (_idx.bytes()) {
        auto tmp = (decltype(_host_idx)) malloc(_idx.bytes());
        _idx.host(tmp);
        _host_idx = tmp;
    }
    if (clear) clearDevice();
    af::sync();
}

void Column::clearDevice() {
    _device = array();
    _idx = array();
}

Column Column::select(af::array const &idx, bool const flush) const {
    Column out(*this);
    if (_type == STRING) {
        out._idx = out._idx(af::span, idx);
        if (flush) out.flush();
        return out;
    }
    out._device = out._device(af::span, idx); //todo change to column wise
}

