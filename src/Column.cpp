#include "include/Column.h"
#include "include/Utils.h"
#include "include/BatchFunctions.h"
#include "include/AFTypes.h"
#include "include/KernelInterface.h"
#include <exception>
#include <cstring>
#define GC_BUFFER 1000000000

std::unordered_map<af::dtype, DataType> Column::_typeMap({{u8, UCHAR}, // NOLINT(cert-err58-cpp)
                                                            {b8, BOOL},
                                                            {u16, USHORT},
                                                            {s16, SHORT},
                                                            {u32, UINT},
                                                            {s32, INT},
                                                            {u64, ULONG},
                                                            {s64, LONG},
                                                            {f32, FLOAT},
                                                            {f64, DOUBLE}});

Column::Column(af::array const &data, DataType const type) : _device(data), _type(type) {
    if (type == STRING) {
        _device = af::flat(_device);
        _generateStringIndex();
    } else if (type != DATE && type != DATETIME && type != TIME) {
        _device = Utils::hflat(_device);
    }
}
Column::Column(af::array &&data, DataType const type) : _device(std::move(data)), _type(type) {
    if (type == STRING) {
        _device = af::flat(_device);
        _generateStringIndex();
    } else if (type != DATE && type != DATETIME && type != TIME) {
        _device = Utils::hflat(_device);
    }
}
Column::Column(af::array const &data) : _type(_typeMap.at(data.type())), _device(data) {
    _device = Utils::hflat(_device);
}
Column::Column(af::array &&data) : _type(_typeMap.at(std::move(data).type())), _device(data) {
    _device = Utils::hflat(_device);
}
Column::Column(af::array const &data, af::array const &index) : _device(data), _idx(index) {
    _device = af::flat(_device);
}
Column::Column(af::array &&data, af::array &&index) : _device(std::move(data)), _idx(std::move(index)) {
    _device = af::flat(_device);
}
Column::Column::Column(Column &&other) noexcept :  _device(std::move(other._device)), _idx(std::move(other._idx)),
    _type(other._type), _host(other._host), _host_idx(other._host_idx) {
    other._host_idx = nullptr;
    other._host = nullptr;
}
Column::Column(Column::Proxy &&data, Column::Proxy &&idx) {
    _device = data;
    _device = af::flat(_device);
    _idx = idx;
    _device.eval();
    _idx.eval();
}
Column::Column(Column::Proxy const &data, Column::Proxy const &idx) {
    _device = data;
    _device = af::flat(_device);
    _idx = idx;
    _device.eval();
    _idx.eval();
}
Column::Column(Column::Proxy &&data, DataType const type) : _type(type) {
    _device = data;
    if (type == STRING) {
        _device = af::flat(_device);
        _generateStringIndex();
    } else if (type != DATE && type != DATETIME && type != TIME) {
        _device = Utils::hflat(_device);
    }
    _device.eval();
}
Column::Column(Column::Proxy const &data, DataType const type) : _type(type) {
    _device = data;
    if (type == STRING) {
        _device = af::flat(_device);
        _generateStringIndex();
    } else if (type != DATE && type != DATETIME && type != TIME) {
        _device = Utils::hflat(_device);
    }
    _device.eval();
}
Column::Column(Proxy const &data) : _type(_typeMap.at(data.type())) {
    _device = data;
    _device = Utils::hflat(_device);
    _device.eval();
}
Column::Column(Proxy &&data) : _type(_typeMap.at(std::move(data).type())) {
    _device = data;
    _device = Utils::hflat(_device);
    _device.eval();
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
        output(k, b) = Utils::hflat(flat(output(k, b)) | flat(_device(_idx(0, b) + i) << j));
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

af::array Column::operator==(Column const &other) {
    if (_type != other._type) { throw std::runtime_error("Mismatch column type"); }
    if (length() != other.length()) { throw std::runtime_error("Mismatch column length"); }
    if (_type == STRING || _type == DATE || _type == TIME || _type == DATETIME) {
        auto b =  hash(false) == other.hash(false);
        if (_type != STRING) return b;
        if (where(b).isempty()) return b;
        return stringComp(_device, other._device, _idx(af::span, b), other._idx(af::span, b));
    }
    return _device == other._device;
}

af::array Column::operator!= (Column const &other) { return !(*this == other); }

#define ASSIGN(OP) \
af::array Column::operator OP(Column const &other) { \
    if (_type != other._type) { throw std::runtime_error("Mismatch column type"); } \
    if (_type == STRING) { throw std::runtime_error("Invalid column type"); } \
    if (length() != other.length()) { throw std::runtime_error("Mismatch column length"); } \
    if (_type == DATE || _type == TIME || _type == DATETIME) { \
        return hash() OP other.hash(); \
    } \
    return _device OP other._device; \
}
ASSIGN(<)
ASSIGN(>)
ASSIGN(<=)
ASSIGN(>=)
#undef ASSIGN
#define ASSIGN(OP) \
Column Column::operator OP(Column const &other) { \
    if (_type == STRING || _type == DATE || _type == TIME || _type == DATETIME || \
        other._type == STRING || other._type == DATE || other._type == TIME || other._type == DATETIME) { \
        throw std::runtime_error("Operation on invalid column type"); \
    } \
    return Column(_device OP other._device); \
}
ASSIGN(+)
ASSIGN(-)
ASSIGN(*)
ASSIGN(/)
#undef ASSIGN


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

void Column::toHost() {
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
    clearDevice();
    af::sync();
}

void Column::clearDevice() {
    static size_t bytes_cleared = 0;
    bytes_cleared += _device.bytes();
    bytes_cleared += _idx.bytes();
    _device = af::array();
    _idx = af::array();
    if (bytes_cleared > GC_BUFFER) af::deviceGC();
}

Column Column::select(af::array const &rows) const {
    if (_type == STRING) {
        af::array idx = _idx(af::span, rows);
        return Column(stringGather(_device, idx), idx);
    }
    return Column(_device(af::span, rows), _type);
}

af::array Column::_dehashDate(af::array const &key, DateFormat const dateFormat) {
    switch (dateFormat) {
        case YYYYMMDD:
            return join(0, key / 10000, (key / 100) % 100, key % 100).as(u16);
        case YYYYDDMM:
            return join(0, key / 10000, key % 100, (key / 100) % 100).as(u16);
        case MMDDYYYY:
            return join(0, key % 10000, key / 1000000, (key / 10000) % 100).as(u16);
        case DDMMYYYY:
            return join(0, key % 10000, (key / 10000) % 100, key / 1000000).as(u16);
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

Column Column::left(unsigned int length) const {
    if (type() != STRING) throw std::runtime_error("Expected String");
    if (length == 0) throw std::invalid_argument("Must be > 0");
    auto len = length + 1;
    auto idx = _idx;
    idx(1, idx.row(1) > len) = len;
    return Column(stringGather(_device, idx), idx);
}

Column Column::right(unsigned int length) const {
    if (type() != STRING) throw std::runtime_error("Expected String");
    if (length == 0) throw std::invalid_argument("Must be > 0");
    auto len = length + 1;
    auto idx = _idx;
    auto b = idx.row(1) > len;
    idx(0, b) = idx(1, b) - len;
    idx(1, b) = len;
    return Column(stringGather(_device, idx), idx);
}

Column Column::trim(unsigned int start, unsigned int length) const {
    if (type() != STRING) throw std::runtime_error("Expected String");
    if (start == 0 || length == 0) throw std::invalid_argument("start and length must be > 0");
    auto len = length + 1;
    auto idx = _idx;
    auto b = idx.row(1) > len;
    idx(0, b) += start;
    idx(1, b) = len;
    return Column(stringGather(_device, idx), std::move(idx));
}

void Column::_generateStringIndex() {
    using namespace Utils;
    _idx = af::diff1(af::join(1, af::constant(0, 1, u64), hflat(where64(_device == 0))), 1);
    _idx(0) += 1;
    _idx = _idx.elements() == 1 ? af::join(0, af::constant(0, 1, _idx.type()), _idx) :
            af::join(0, af::scan(_idx, 1, AF_BINARY_ADD, false), _idx);
}

template<typename T>
void Column::cast() {
    using namespace Utils;
    if (_type == DATE || _type == TIME || _type == DATETIME) throw std::runtime_error("Invalid Type");
    if (_type == STRING) {
        _device = _device(_device != ' ');
        _generateStringIndex();
        _device = numericParse<T>(_device, _idx);
    } else {
        _device = _device.as(GetAFType<T>().af_type);
    }
    _type = GetAFType<T>().df_type;
    _idx = af::array(0, u64);
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

af::array operator==(char const* lhs, Column const &rhs) {
    if (rhs._type != STRING) throw std::runtime_error("Type mismatch");
    return stringComp(rhs._device, lhs, rhs._idx);
}
af::array operator==(Column const &lhs, char const* rhs) { return rhs == lhs; }
af::array operator!=(char const* lhs, Column const &rhs) { return !(lhs == rhs); }
af::array operator!=(Column const &lhs, char const* rhs) { return !(rhs == lhs); }

#define ASSIGN(OP) \
template<typename T> \
af::array operator OP(T const &lhs, Column const &rhs) { \
    if (rhs._type == STRING || rhs._type == DATE || rhs._type == TIME || rhs._type == DATETIME) { \
        throw std::runtime_error("Operation on invalid column type"); \
    } \
    return lhs OP rhs._device; \
} \
template<typename T> \
af::array operator OP(Column const &lhs, T const &rhs) { return rhs OP lhs; }
ASSIGN(==)
ASSIGN(!=)
ASSIGN(>=)
ASSIGN(<=)
ASSIGN(>)
ASSIGN(<)
#undef ASSIGN

#define ASSIGN(OP) \
template<typename T> \
Column operator OP(T const &lhs, Column const &rhs) { \
    if (rhs._type == STRING || rhs._type == DATE || rhs._type == TIME || rhs._type == DATETIME) { \
        throw std::runtime_error("Operation on invalid column type"); \
    } \
    return Column(lhs OP rhs._device); \
} \
template<typename T> \
Column operator OP(Column const &lhs, T const &rhs) { return rhs OP lhs; }
ASSIGN(+)
ASSIGN(-)
ASSIGN(*)
ASSIGN(/)
#undef ASSIGN

#define ASSIGN(TP) \
template af::array operator== <TP>(Column const &lhs, TP const &rhs); \
template af::array operator!= <TP>(TP const &lhs, Column const &rhs); \
template af::array operator>= <TP>(Column const &lhs, TP const &rhs); \
template af::array operator<= <TP>(TP const &lhs, Column const &rhs); \
template af::array operator> <TP>(Column const &lhs, TP const &rhs); \
template af::array operator< <TP>(TP const &lhs, Column const &rhs); \
template Column operator+ <TP>(Column const &lhs, TP const &rhs); \
template Column operator- <TP>(TP const &lhs, Column const &rhs); \
template Column operator* <TP>(Column const &lhs, TP const &rhs); \
template Column operator/ <TP>(TP const &lhs, Column const &rhs);
ASSIGN(float)
ASSIGN(double)
ASSIGN(char)
ASSIGN(unsigned char)
ASSIGN(short)
ASSIGN(unsigned short)
ASSIGN(int)
ASSIGN(unsigned int)
ASSIGN(long long)
ASSIGN(unsigned long long)
ASSIGN(bool)
#undef ASSIGN