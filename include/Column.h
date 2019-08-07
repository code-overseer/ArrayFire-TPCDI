#ifndef ARRAYFIRE_TPCDI_COLUMN_H
#define ARRAYFIRE_TPCDI_COLUMN_H
#include <arrayfire.h>
#include <unordered_map>
#include <utility>
#include "Enums.h"
#include "AFTypes.h"

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
    size_t _length = 0;
public:
    Column(af::array const &data, DataType const type) : _device(data), _type(type) { _length = _device.dims(1); }
    Column(af::array const &data, af::array const &index) : _device(data), _idx(index) { _length = _idx.dims(1); }
    Column(af::array &&data, DataType const type) : _device(std::move(data)), _type(type) { _length = _device.dims(1); }
    Column(af::array &&data, af::array &&index) : _device(std::move(data)), _idx(std::move(index)) { _length = _idx.dims(1); }
    Column(Column &&other) noexcept;
    Column(Proxy &&data, DataType type);
    Column(Proxy const &data, DataType type);
    Column(Proxy &&data, Proxy &&idx);
    Column(Proxy const &data, Proxy const &idx);
    Column(Column const &other) = default;
    virtual ~Column() { if (_host) free(_host); if (_host_idx) free(_host_idx); }
    Column& operator=(Column &&other) noexcept;
    Column& operator=(Column const &other) = default;
    Column concatenate(Column &bottom);
    Column select(af::array const &rows) const;
    void flush();
    void toDate();
    void toDate(bool isDelimited, DateFormat dateFormat = YYYYMMDD);
    void toTime(bool isDelimited);
    void toTime();
    void toDateTime(bool isDelimited, DateFormat dateFormat = YYYYMMDD);
    af::array hash(bool sortable = false) const;
    void toHost(bool clear = false);
    void clearDevice();
    void printColumn();
    static af::array endDate() {
        return join(0, af::constant(9999, 1, u16), af::constant(12, 1, u16), af::constant(31, 1, u16));
    }
    af::array left(unsigned int length);
    af::array right(unsigned int length);
    Column trim(unsigned int start, unsigned int length);
    template<typename T> void cast();
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
    inline af::array& operator()() { return _device; }
    inline size_t length() { return _length; }

    #define ASSIGN(OP) \
    af::array operator OP(Column const &other); \
    af::array operator OP(af::array const &other); \
    af::array operator OP(af::array &&other);
    ASSIGN(==)
    ASSIGN(!=)
    ASSIGN(>=)
    ASSIGN(<=)
    ASSIGN(>)
    ASSIGN(<)
    #undef ASSIGN
    af::array operator ==(char const *other);

};

#endif //ARRAYFIRE_TPCDI_COLUMN_H
