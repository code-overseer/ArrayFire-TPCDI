#include "include/Column.h"
#include "include/TPCDI_Utils.h"
#include "include/BatchFunctions.h"
#include <exception>
#include <cstring>
#ifdef USING_OPENCL
    #include "include/OpenCL/opencl_parsers.h"
#elif defined(USING_CUDA)
    #include "include/CUDA/cuda_kernels.h"
#else
    #include "include/CPU/vector_functions.h"
#endif

Column::Column(Column &&other) noexcept :  _device(std::move(other._device)), _idx(std::move(other._idx)),
    _type(other._type), _host(other._host), _host_idx(other._host_idx) {
    other._length = 0;
    other._host_idx = nullptr;
    other._host = nullptr;
}

Column::Column(Column::Proxy &&data, Column::Proxy &&idx) {
    _device = data; _idx = idx;
}

Column::Column(Column::Proxy const &data, Column::Proxy const &idx) {
    _device = data; _idx = idx; _length = _idx.dims(1);
}

Column::Column(Column::Proxy &&data, DataType const type) : _type(type) {
    _device = data; _length = _device.dims(1);
}

Column::Column(Column::Proxy const &data, DataType const type) : _type(type) {
    _device = data; _length = _device.dims(1);
}

Column &Column::operator=(Column &&other) noexcept {
    _device = std::move(other._device);
    _idx = std::move(other._idx);
    _type = other._type;
    _length = other._length;
    _host = other._host;
    _host_idx = other._host_idx;
    other._length = 0;
    other._host = nullptr;
    other._host_idx = nullptr;
    return *this;
}

af::array Column::_fnv1a() const {
    using namespace af;
    auto const loop_length = sum<unsigned long long>(max(_idx.row(1), 1));
    auto const offset = 0xcbf29ce484222325llU;
    auto const prime = 0x100000001b3llU;
    af::array output = constant(offset, dim4(1, _idx.elements() / 2), u64);
    for (ull i = 0; i < loop_length; ++i) {
        auto b = _idx.row(1) > i;
        auto low8 = flat((output(b) & 0xffllU)) ^ _device(_idx(0, b) + i);
        output(b) = (flat(output(b) & ~0xffllU) | low8);
        output(b) *= prime;
    }
    output.eval();
    return output;
}

af::array Column::_wordHash() const {
    using namespace af;
    auto const loop_length = sum<unsigned long long>(max(_idx.row(1), 1));
    auto const output_width = loop_length / 8 + ((loop_length % 8) > 0);
    af::array output = constant(0, dim4(output_width, _idx.elements() / 2), u64);
    int k = 0;
    for (ull i = 0; i < loop_length; ++i) {
        auto b = _idx.row(1) > i;
        auto j = (7 - (i % 8)) * 8;
        output(k, b) = TPCDI_Utils::hflat(flat(output(k, b)) | flat(_device(_idx(0, b) + i) << j));
        if (!j) ++k;
    }
    output.eval();
    return output;
}

af::array Column::_dateHash() const {
    auto mult = af::flip(pow(100Ull, af::range(af::dim4(3), 0, u64)), 0);
    auto key = batchFunc(mult, _device, BatchFunctions::batchMult);
    return sum(key, 0).as(u64);
}

af::array Column::_datetimeHash() const {
    auto mult = af::flip(pow(100Ull, af::range(af::dim4(6), 0, u64)), 0);
    auto key = batchFunc(mult, _device, BatchFunctions::batchMult);
    return sum(key, 0).as(u64);
}

af::array Column::hash(bool const sortable) const {
    if (_type == STRING) return sortable ? _wordHash() : _fnv1a();
    if (_type == DATE) return _dateHash();
    if (_type == TIME) return _timeHash();
    if (_type == DATETIME) return _datetimeHash();
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

Column Column::concatenate(Column const &bottom) const {
    using namespace BatchFunctions;
    if (_type != bottom._type) throw std::runtime_error("Type mismatch");
    if (_type == STRING) {
        auto i = bottom._idx;
        i.row(0) = af::batchFunc(i.row(0), af::sum(_idx.col(af::end), 0), batchAdd);
        return Column(join(0, _device, bottom._device), join(1, _idx, i));
    }
    return Column(join(1, _device, bottom._device), _type);
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
    _device = af::array();
    _idx = af::array();
}

Column Column::select(af::array const &rows) const {
    if (_type == STRING) {
        af::array idx = _idx(af::span, rows);
        return Column(stringGather(_device, idx), idx);
    }
    return Column(_device(af::span, rows), _type);
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

af::array Column::_dehashDate(af::array const &key, DateFormat const dateFormat) {
    switch (dateFormat) {
        case YYYYMMDD:
            return join(0, _device / 10000, (_device / 100) % 100, _device % 100).as(u16);
        case YYYYDDMM:
            return join(0, _device / 10000, _device % 100, (_device / 100) % 100).as(u16);
        case MMDDYYYY:
            return join(0, _device % 10000, _device / 1000000, (_device / 10000) % 100).as(u16);
        case DDMMYYYY:
            return join(0, _device % 10000, (_device / 10000) % 100, _device / 1000000).as(u16);
        default:
            throw std::runtime_error("No such date format");
    }
}

void Column::toDate(bool const isDelimited, DateFormat const dateFormat) {
    using namespace af;
    using namespace BatchFunctions;
    if (_type != STRING) throw std::runtime_error("Expected String type");
    if (isDelimited) _device = _device((_device >= '0' && _device <= '9') || _device == 0);
    cast<unsigned int>();
    _type = DATE;
    _device = _dehashDate(_device, dateFormat);
    _device.eval();
}

void Column::toTime(bool const isDelimited) {
    using namespace af;
    using namespace BatchFunctions;
    if (_type != STRING) throw std::runtime_error("Expected String type");
    if (isDelimited) _device = _device((_device >= '0' && _device <= '9') || _device == 0);
    cast<unsigned int>();
    _type = TIME;
    _device = _dehashTime(_device);
    _device.eval();
}

void Column::toDateTime(DateFormat const dateFormat) {
    using namespace af;
    using namespace BatchFunctions;
    if (_type != STRING) throw std::runtime_error("Expected String type");
    _device = _device((_device >= '0' && _device <= '9') || _device == 0);
    cast<unsigned long long>();
    _type = DATETIME;
    _device = join(0, _dehashDate(_device / 1000000, dateFormat), _dehashTime(_device % 1000000));
    _device.eval();
}

void Column::printColumn() const {
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

af::array Column::left(unsigned int length) const {
    if (type() != STRING) throw std::runtime_error("Expected String");
    if (!where(af::anyTrue(_idx(1, af::span) < length)).isempty()) throw std::runtime_error("Some strings are too short");
    auto idx = _idx;
    idx.row(1) = length;
    auto out = stringGather(_device, idx);
    return moddims(out, af::dim4(length, out.elements() / length));
}
af::array Column::right(unsigned int length) const {
    if (type() != STRING) throw std::runtime_error("Expected String");
    if (!where(af::anyTrue(_idx(1, af::span) < length + 1)).isempty()) throw std::runtime_error("Some strings are too short");
    auto end = af::sum(_idx.as(s64), 0) - 2;
    auto out = af::batchFunc(end, af::range(af::dim4(length), s32), BatchFunctions::batchSub);
    out = moddims(_device(out), out.dims());
    return out;
}

Column Column::trim(unsigned int start, unsigned int length) const {
    if (type() != STRING) throw std::runtime_error("Expected String");
    if (!where(anyTrue(flat(_idx(1, af::span) <= (length + start)))).isempty()) throw std::runtime_error("Some strings are too short");
    auto idx = _idx;
    idx.row(1) = length + 1;
    auto out = Column(stringGather(_device, idx), std::move(idx));
    out._device(out.irow(0) + length) = 0;
    return out;
}

template<typename T>
void Column::cast() {
    using namespace TPCDI_Utils;
    if (_type == DATE || _type == TIME || _type == DATETIME) throw std::runtime_error("Invalid Type");
    if (_type == STRING) {
        _device = _device(_device != ' ');
        _idx = af::diff1(af::join(1, af::constant(0, 1, u64), hflat(where64(_device == 0))), 1);
        _idx(0) += 1;
        _idx = join(0, af::scan(_idx, 1, AF_BINARY_ADD, false), _idx);
        _device = numericParse<T>(_device, _idx);
    } else {
        _device = _device.as(GetAFType<T>().af_type);
    }
    _type = GetAFType<T>().df_type;
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