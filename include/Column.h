#ifndef ARRAYFIRE_TPCDI_COLUMN_H
#define ARRAYFIRE_TPCDI_COLUMN_H
#include <arrayfire.h>
#include <unordered_map>
#include <utility>
#include "Enums.h"
#include "AFTypes.h"

/* Wrapper for array to simplify access for different types (especially strings) */
class Column {
    af::array _device;
    af::array _idx;
    void* _host = nullptr;
    unsigned long long* _host_idx = nullptr;
    DataType _type = FLOAT;
public:
    Column(af::array data, af::array idx, DataType const type) : _device(std::move(data)), _idx(std::move(idx)), _type(type) {}
    Column(af::array &&data, af::array &&idx, DataType const type) : _device(std::move(data)), _idx(std::move(idx)), _type(type) {}
    inline af::array& index() { return _idx; }
    inline af::array& array() { return _device; }
    inline DataType type() { return _type; }
    inline DataType type(DataType type) { _type = type; return _type; }
    af::array::array_proxy& operator()(af::index const &x, af::index const &y);
    af::array hash(bool sortable = false) const;
    void refresh();

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
