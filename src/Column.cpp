#include "include/Column.h"
#include "include/TPCDI_Utils.h"
#include "include/BatchFunctions.h"
#include <exception>
#include <cstring>
#ifdef USING_OPENCL
    #include "include/OpenCL/opencl_parsers.h"
#else
    #include "include/CPU/vector_functions.h"
#endif


Column::Column(Column &&other) noexcept :  _device(std::move(other._device)), _idx(std::move(other._idx)),
    _type(other._type), _host(other._host), _host_idx(other._host_idx) {
    other._host_idx = nullptr;
    other._host = nullptr;
}

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
    if (_type == STRING || _type == DATE || _type == TIME || _type == DATETIME) { \
        auto lhs = hash(false); \
        return lhs == other; \
    } \
    return _device == other; \
} \
af::array Column::operator!=(af::array OP other) { return !(*this == other); }
ASSIGN( const &)
ASSIGN(&&)

#undef ASSIGN
#define ASSIGN(OP) \
af::array Column::operator OP(af::array const &other) { \
    af::array lhs; \
    if (_type == STRING) { throw std::runtime_error("Invalid column type"); } \
    if (_type == DATE || _type == TIME || _type == DATETIME) { \
        lhs = hash(false); \
        return lhs OP other; \
    } \
    return _device OP other; \
} \
af::array Column::operator OP(af::array &&other) { return *this OP other; }
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
    _device = stringGather(_device, _idx);
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
    flush();
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
    _device = af::array();
    _idx = af::array();
}

Column Column::select(af::array const &idx) const {
    Column out(*this);
    if (_type == STRING) {
        out._idx = out._idx(af::span, idx);
        out.flush();
    } else {
        out._device = out._device(af::span, idx); //todo change to column wise
    }
    return out;
}

af::array Column::operator==(char const *other) {
    if (_type != STRING) throw std::runtime_error("Type mismatch");
    size_t const length = strlen(other) + 1;
    using namespace af;
    auto rhs = af::array(length, other).as(u8);
    auto out = _idx.row(1) == length;
    auto lhs = _device(batchFunc(_idx(0, out), range(dim4(length),0, u32), BatchFunctions::batchAdd));
    lhs = moddims(lhs, dim4(length, lhs.elements() / length));
    out = allTrue(batchFunc(lhs, rhs, BatchFunctions::batchEqual), 0);
    return out;
}

void Column::toDate(bool const isDelimited, DateFormat const dateFormat) {
    using namespace af;
    if (_type != STRING) throw std::runtime_error("Expected String type");
    _type = DATE;
    _device = TPCDI_Utils::stringToDate(_device, isDelimited, dateFormat);
    _idx = array(0, u64);
    _device.eval();
}

void Column::toTime(bool const isDelimited) {
    using namespace af;
    using namespace BatchFunctions;
    if (_type != STRING) throw std::runtime_error("Expected String type");
    _type = TIME;
    _device = TPCDI_Utils::stringToTime(_device, isDelimited);
    _idx = array(0, u64);
    _device.eval();
}

void Column::toDateTime(bool const isDelimited, DateFormat const dateFormat) {
    using namespace af;
    using namespace BatchFunctions;
    if (_type != STRING) throw std::runtime_error("Expected String type");
    _type = DATETIME;
    _device = TPCDI_Utils::stringToDateTime(_device, isDelimited, dateFormat);
    _idx = array(0, u64);
    _device.eval();
}

void Column::print() {
    if (_type != STRING) {
        af_print(_device);
    } else {
        printStr(_device);
    }
}

void Column::toDate() {
    if (_type != DATETIME) throw std::runtime_error("Expected DateTime");
    _type = DATE;
    _device = _device(af::seq(3), af::span);
    _device.eval();
}

void Column::toTime() {
    if (_type != DATETIME) throw std::runtime_error("Expected DateTime");
    _type = TIME;
    _device = _device(af::range(af::dim4(3)) + 3, af::span);
    _device.eval();
}

af::array Column::left(unsigned int length) {
    if (type() != STRING) throw std::runtime_error("Expected String");
    if (!where(anyTrue(_idx(1, af::span) < length)).isempty()) throw std::runtime_error("Some strings are too short");
    auto out = af::batchFunc(_idx(0, af::span), af::range(af::dim4(length), u32), BatchFunctions::batchAdd);
    out = moddims(_device(out), out.dims());
    return out;
}
af::array Column::right(unsigned int length) {
    if (type() != STRING) throw std::runtime_error("Expected String");
    if (!where(anyTrue(_idx(1, af::span) < length + 1)).isempty()) throw std::runtime_error("Some strings are too short");
    auto end = af::accum(_idx(1, af::span)).as(s64) - 2;
    auto out = af::batchFunc(end, af::range(af::dim4(length), s32), BatchFunctions::batchSub);
    out = moddims(_device(out), out.dims());
    return out;
}

Column Column::trim(unsigned int start, unsigned int length) {
    if (type() != STRING) throw std::runtime_error("Expected String");
    if (!where(anyTrue(_idx(1, af::span) < (start + length))).isempty())
        throw std::runtime_error("Some strings are too short");
    auto out = af::batchFunc(_idx(0, af::span), af::range(af::dim4(length + 1), u32) + start, BatchFunctions::batchAdd);
    out.row(af::end) = _device.elements() - 1;

    return Column(_device(out), STRING);
}

template<typename T>
void Column::cast() {
    using namespace TPCDI_Utils;
    if (_type == DATE || _type == TIME || _type == DATETIME) throw std::runtime_error("Invalid Type");
    if (_type == STRING) {
        _device = _device(_device != ' ');
        _idx = af::diff1(af::join(1, af::constant(0, 1, u64), flipdims(where64(_device == 0))), 1);
        _idx(0) += 1;
        _idx = join(0, af::scan(_idx, 1, AF_BINARY_ADD, false), _idx);
        af::array out;
        numericParse<T>(out, _device, _idx);
        _device = std::move(out);
    } else {
        _device = _device.as(GetAFType<T>().af_type);
        _type = GetAFType<T>().df_type;
    }
}
template void Column::cast<unsigned char>();
template void Column::cast<short>();
template void Column::cast<unsigned short>();
template void Column::cast<int>();
template void Column::cast<unsigned int>();
template void Column::cast<long long>();
template void Column::cast<double>();
template void Column::cast<float>();
template void Column::cast<unsigned long long>();