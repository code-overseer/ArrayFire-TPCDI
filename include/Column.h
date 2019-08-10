#ifndef ARRAYFIRE_TPCDI_COLUMN_H
#define ARRAYFIRE_TPCDI_COLUMN_H
#include <arrayfire.h>
#include <unordered_map>
#include <utility>
#include "Enums.h"

/* Wrapper for array to simplify access for different types (especially strings) */
class Column {
    typedef af::array::array_proxy Proxy;
    af::array _device;
    af::array _idx = af::array(0,u64);
    void* _host = nullptr;
    unsigned long long* _host_idx = nullptr;
    DataType _type = STRING;
    af::array _fnv1a() const;
    af::array _wordHash() const;
    af::array _datetimeHash() const;
    af::array _dateHash() const;
    inline af::array _timeHash() const { return _dateHash(); }
    static af::array _dehashDate(af::array const &key, DateFormat format);
    static af::array _dehashTime(af::array const &key) { return join(0, key / 10000, key / 100 % 100, key % 100).as(u16); }
    void _generateStringIndex();
    static std::unordered_map<af::dtype, DataType> _typeMap;
public:
    Column(af::array const &data, DataType type);
    Column(af::array &&data, DataType type);
    explicit Column(af::array const &data);
    explicit Column(af::array &&data);
    Column(af::array const &data, af::array const &index);
    Column(af::array &&data, af::array &&index);
    Column(Column &&other) noexcept;
    Column(Proxy &&data, DataType type);
    Column(Proxy const &data, DataType type);
    Column(Proxy &&data, Proxy &&idx);
    Column(Proxy const &data, Proxy const &idx);
    explicit Column(Proxy const &data);
    explicit Column(Proxy &&data);
    Column(Column const &other) = default;
    virtual ~Column() { if (_host) free(_host); if (_host_idx) free(_host_idx); }
    Column& operator=(Column &&other) noexcept;
    Column& operator=(Column const &other) = default;
    Column concatenate(Column const &bottom) const;
    Column select(af::array const &rows) const;
    void toDate();
    void toTime();
    void toDate(bool isDelimited, DateFormat dateFormat = YYYYMMDD);
    void toTime(bool isDelimited);
    void toDateTime(DateFormat dateFormat = YYYYMMDD);
    void toHost();
    void clearDevice();
    template<typename T> void cast();
    af::array hash(bool sortable = false) const;
    void printColumn() const;
    Column left(unsigned int length) const;
    Column right(unsigned int length) const;
    Column trim(unsigned int start, unsigned int length) const;
    inline af::array const& index() const { return _idx; }
    inline af::array const& data() const { return _device; }
    inline af::dim4 dims() const { return _device.dims(); }
    inline bool isempty() const { return _device.isempty(); }
    inline dim_t dims(unsigned int const i) const { return _device.dims(i); }
    inline DataType type() const { return _type; }
    inline DataType type(DataType const type) { _type = type; return _type; }
    inline Proxy row(int const i) const { return _device.row(i); }
    inline Proxy col(int const i) const { return _device.col(i); }
    inline Proxy rows(int i, int j) const { return _device.rows(i, j); }
    inline Proxy cols(int i, int j) const { return _device.cols(i, j); }
    inline Proxy irow(int const i) const { return _idx.row(i); }
    inline Proxy icol(int const i) const { return _idx.col(i); }
    inline Proxy irows(int i, int j) const { return _idx.rows(i, j); }
    inline Proxy icols(int i, int j) const { return _idx.cols(i, j); }
    inline Proxy index(af::index const &x) const { return _idx(x); }
    inline Proxy index(af::index const &x, af::index const &y) const { return _idx(x, y); }
    inline Proxy operator()(af::index const &x) const { return _device(x); }
    inline Proxy operator()(af::index const &x, af::index const &y) const { return _device(x, y); }
    inline size_t length() const { return (_type == STRING) ? _idx.dims(1) : _device.dims(1); }

    #define ASSIGN(OP) \
    af::array operator OP (Column const &other);
    ASSIGN(==)
    ASSIGN(!=)
    ASSIGN(>=)
    ASSIGN(<=)
    ASSIGN(>)
    ASSIGN(<)
    #undef ASSIGN
    #define ASSIGN(OP) \
    Column operator OP (Column const &other);
    ASSIGN(+)
    ASSIGN(-)
    ASSIGN(*)
    ASSIGN(/)
    #undef ASSIGN

    #define ASSIGN(OP) \
    template<typename T> \
    friend af::array operator OP(T const &lhs, Column const &rhs); \
    template<typename T> \
    friend af::array operator OP(Column const &lhs, T const &rhs);
    ASSIGN(==)
    ASSIGN(!=)
    ASSIGN(>=)
    ASSIGN(<=)
    ASSIGN(>)
    ASSIGN(<)
    #undef ASSIGN

    #define ASSIGN(OP) \
    template<typename T> \
    friend Column operator OP(T const &lhs, Column const &rhs); \
    template<typename T> \
    friend Column operator OP(Column const &lhs, T const &rhs);
    ASSIGN(+)
    ASSIGN(-)
    ASSIGN(*)
    ASSIGN(/)
    #undef ASSIGN
    friend af::array operator ==(char const* lhs, Column const &rhs);
    friend af::array operator ==(Column const &lhs, char const*rhs);
    friend af::array operator !=(char const* lhs, Column const &rhs);
    friend af::array operator !=(Column const &lhs, char const*rhs);
};

#endif //ARRAYFIRE_TPCDI_COLUMN_H
