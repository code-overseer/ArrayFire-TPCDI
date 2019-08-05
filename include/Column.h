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
    af::array _idx;
    void* _host = nullptr;
    unsigned long long* _host_idx = nullptr;
    DataType _type = FLOAT;
public:
    Column(af::array const &data, DataType const type) : _device(data), _idx(af::array(0,u64)), _type(type) {}
    Column(af::array &&data, DataType const type) : _device(std::move(data)), _idx(af::array(0,u64)), _type(type) {}
    Column(af::array const &data, af::array const &index) : _device(data), _idx(index), _type(STRING) {}
    Column(af::array &&data, af::array &&index) : _device(std::move(data)), _idx(std::move(index)), _type(STRING) {}
    Column(Column &&other) noexcept :  _device(std::move(other._device)), _idx(std::move(other._idx)),
    _type(other._type), _host(other._host), _host_idx(other._host_idx) {}
    Column(Proxy &&data, Proxy &&idx, DataType type);
    Column(Proxy const &data, Proxy const &idx, DataType type);
    Column(Column const &other) = default;
    Column& operator=(Column &&other) noexcept;
    Column& operator=(Column const &other) = default;
    Column concatenate(Column &bottom);
    Column select(af::array const &idx, bool flush = false) const;
    void flush();
    virtual ~Column() { if (_host) free(_host); if (_host_idx) free(_host_idx); }
    inline af::array& index() { return _idx; }
    inline af::array& array() { return _device; }
    inline af::dim4 dims() const { return _device.dims(); }
    inline bool isempty() const { return _device.isempty(); }
    inline dim_t dims(unsigned int const i) const { return _device.dims(i); }
    inline DataType type() const { return _type; }
    inline DataType type(DataType const type) { _type = type; return _type; }
    inline Proxy row(int i) const { return _device.row(i); }
    inline Proxy col(int i) const { return _device.col(i); }
    inline Proxy rows(int i, int j) const { return _device.rows(i, j); }
    inline Proxy cols(int i, int j) const { return _device.cols(i, j); }
    inline Proxy index(af::index const &x, af::index const &y) const { return _idx(x, y); }
    inline Proxy operator()(af::index const &x, af::index const &y) const { return _device(x, y); }
    af::array hash(bool sortable = false) const;
    void toHost(bool clear = false);
    void clearDevice();

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

};

#endif //ARRAYFIRE_TPCDI_COLUMN_H
